import { AgentsDataSource } from '../infrastructure/data/AgentsDataSource.js';
import { AgentsRepository } from '../domain/agents/repositories/AgentsRepository.js';
import { AgentsService } from '../domain/agents/services/AgentsService.js';
import { ProjectCard } from '../presentation/components/ProjectCard.js';
import { ModalWindow } from '../presentation/components/ModalWindow.js';
import { ErrorHandler, createErrorUI } from '../infrastructure/error/ErrorHandler.js';

/**
 * Agents Controller - Application Layer
 * Координирует взаимодействие между всеми слоями приложения для Agents
 */
export class AgentsController {
    constructor(config) {
        this.config = config;
        this.githubConfig = {
            githubRepo: config.githubRepo,
            githubBranch: config.githubBranch
        };
        
        // Инициализируем слои архитектуры
        this.dataSource = new AgentsDataSource(this.githubConfig);
        this.repository = new AgentsRepository(this.dataSource);
        this.service = new AgentsService(this.repository);
        
        // DOM элементы
        this.contentElement = document.querySelector('.content');
        this.modalElement = document.getElementById('markdown-modal');
        this.searchInput = document.querySelector('.nav-content .search-bar input');
        this.searchButton = document.querySelector('.nav-content .search-bar button');
        this.backToTopButton = document.getElementById('back-to-top');
        
        // Компоненты презентации
        this.modal = new ModalWindow(this.modalElement, this.service);
        this.projectCards = new Map(); // Хранилище карточек проектов
        
        // Состояние приложения
        this.currentFilter = null;
        this.currentSearchQuery = '';
        
        this._initializeEventListeners();
    }

    /**
     * Инициализирует приложение
     */
    async initialize() {
        try {
            // Загружаем данные и рендерим интерфейс
            await this._loadAndRenderData();
            // Проверяем URL hash (позволяет открывать #agents/{projectId} напрямую)
            this.modal.checkUrlHash();
            
        } catch (error) {
            console.error('❌ Failed to initialize agents application:', error);
            this._showCriticalError(error);
        }
    }

    /**
     * Загружает и рендерит данные проектов
     */
    async _loadAndRenderData() {
        if (!this.contentElement) {
            throw new Error("Content element not found.");
        }

        // Показываем индикатор загрузки
        const loadingIndicator = this._createLoadingIndicator();
        this.contentElement.appendChild(loadingIndicator);

        try {
            // Получаем данные через сервис
            const projects = await this.service.getAllProjects();

            // Удаляем индикатор загрузки
            loadingIndicator.remove();

            // Очищаем старые секции
            this._clearOldSections();

            // Создаем секцию проектов
            const projectsSection = this._createProjectsSection();
            
            // Создаем карточки синхронно
            for (const project of projects) {
                this._createAndAddProjectCard(projectsSection, project);
            }

            // Обновляем фильтры в боковой панели
            await this._updateSidebarFilters(projects);

        } catch (error) {
            console.error('Error loading agents data:', error);
            
            // Удаляем индикатор загрузки
            loadingIndicator.remove();
            
            // Показываем ошибку
            this._showDataLoadError(error);
        }
    }

    /**
     * Создает и добавляет карточку проекта - СИНХРОННАЯ ВЕРСИЯ
     */
    _createAndAddProjectCard(projectsSection, project) {
        const projectCard = new ProjectCard(project, this.githubConfig, this.dataSource);
        const cardElement = projectCard.createElement(); // Синхронное создание карточки
        
        const weeksGrid = projectsSection.querySelector('.weeks-grid');
        
        if (weeksGrid) {
            weeksGrid.appendChild(cardElement);
            
            // Сохраняем ссылку на карточку для управления
            this.projectCards.set(project.getId(), projectCard);
        } else {
            console.error('❌ Weeks grid not found!');
        }
    }

    /**
     * Создает секцию проектов
     */
    _createProjectsSection() {
        const projectsSection = document.createElement('section');
        projectsSection.id = 'projects';
        projectsSection.className = 'year-section'; // используем тот же класс что и на главной
        projectsSection.innerHTML = `
            <h2 class="year-title section-heading">Agents Projects</h2>
            <div class="weeks-grid"></div>
        `;
        this.contentElement.appendChild(projectsSection);
        return projectsSection;
    }

    /**
     * Обновляет фильтры в боковой панели
     */
    async _updateSidebarFilters(projects) {
        // Обновляем популярные теги
        await this._updatePopularTags();
        
        // Обновляем featured projects
        await this._updateFeaturedProjects();
    }

    /**
     * Обновляет популярные теги
     */
    async _updatePopularTags() {
        try {
            const popularTags = await this.service.getPopularTags(6);
            const tagsContainer = document.querySelector('.tags-container');
            
            if (tagsContainer) {
                tagsContainer.innerHTML = '';
                popularTags.forEach(({ tag, count }) => {
                    const tagElement = document.createElement('span');
                    tagElement.className = 'tag';
                    tagElement.textContent = tag;
                    tagElement.title = `${count} проектов`;
                    
                    tagElement.addEventListener('click', () => {
                        this._filterByTag(tag);
                    });
                    
                    tagsContainer.appendChild(tagElement);
                });
            }
        } catch (error) {
            // Сбой загрузки тегов не критичен
        }
    }

    /**
     * Обновляет featured projects
     */
    async _updateFeaturedProjects() {
        try {
            const featuredProjects = await this.service.getFeaturedProjects(3);
            const featuredList = document.querySelector('.sidebar-section:last-child ul');
            
            if (featuredList) {
                featuredList.innerHTML = '';
                featuredProjects.forEach(project => {
                    const li = document.createElement('li');
                    const a = document.createElement('a');
                    a.href = `#${project.getId()}`;
                    a.textContent = project.title;
                    
                    a.addEventListener('click', (e) => {
                        e.preventDefault();
                        this._openProject(project.getId(), project.title);
                    });
                    
                    li.appendChild(a);
                    featuredList.appendChild(li);
                });
            }
        } catch (error) {
            // Сбой загрузки featured проектов не критичен
        }
    }

    /**
     * Фильтрует карточки по тегу
     */
    _filterByTag(tag) {
        this.currentFilter = { type: 'tag', value: tag };
        this._applyCurrentFilter();
    }

    /**
     * Применяет текущий фильтр
     */
    _applyCurrentFilter() {
        this.projectCards.forEach((projectCard) => {
            let shouldShow = true;
            
            if (this.currentFilter) {
                switch (this.currentFilter.type) {
                    case 'tag':
                        shouldShow = projectCard.hasTag(this.currentFilter.value);
                        break;
                }
            }
            
            // Применяем поиск если есть
            if (shouldShow && this.currentSearchQuery) {
                shouldShow = projectCard.matchesSearch(this.currentSearchQuery);
            }
            
            if (shouldShow) {
                projectCard.show();
                if (this.currentSearchQuery) {
                    projectCard.highlight(this.currentSearchQuery);
                } else {
                    projectCard.removeHighlight();
                }
            } else {
                projectCard.hide();
            }
        });
    }

    /**
     * Выполняет поиск
     */
    async _performSearch(query) {
        if (!query || query.trim().length < 2) {
            this._showSearchModal('Запрос должен содержать минимум 2 символа.');
            return;
        }

        this.currentSearchQuery = query;
        this._applyCurrentFilter();
    }

    /**
     * Открывает проект в модальном окне
     */
    _openProject(projectId, title) {
        // Используем метод модального окна напрямую
        this.modal.open(projectId, projectId, title);
    }

    /**
     * Инициализирует обработчики событий
     */
    _initializeEventListeners() {
        // Поиск
        if (this.searchButton && this.searchInput) {
            this.searchButton.addEventListener('click', () => {
                this._performSearch(this.searchInput.value);
            });
            
            this.searchInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this._performSearch(this.searchInput.value);
                }
            });
        }

        // Кнопка "Наверх"
        if (this.backToTopButton) {
            window.addEventListener('scroll', () => {
                this.backToTopButton.classList.toggle('visible', window.pageYOffset > 300);
            });
            
            this.backToTopButton.addEventListener('click', () => {
                window.scrollTo({ top: 0, behavior: 'smooth' });
            });
        }

        // Обработчики custom events от карточек
        document.addEventListener('projectCardClicked', (event) => {
            const { projectId, title } = event.detail;
            this._openProject(projectId, title);
        });
    }

    /**
     * Создает индикатор загрузки
     */
    _createLoadingIndicator() {
        const loadingIndicator = document.createElement('div');
        loadingIndicator.className = 'loading-indicator';
        loadingIndicator.innerHTML = `
            <div class="loader"></div>
            <p>Загрузка проектов...</p>
        `;
        return loadingIndicator;
    }

    /**
     * Очищает старые секции
     */
    _clearOldSections() {
        this.contentElement.querySelectorAll('.year-section:not(#home)').forEach(section => section.remove());
        this.projectCards.clear();
    }

    /**
     * Показывает критическую ошибку
     */
    _showCriticalError(error) {
        if (!this.contentElement) return;
        
        const criticalError = createErrorUI(
            'unknown',
            'приложение agents',
            () => {
                window.location.reload();
            },
            null
        );
        
        this.contentElement.innerHTML = '';
        this.contentElement.appendChild(criticalError);
    }

    /**
     * Показывает ошибку загрузки данных
     */
    _showDataLoadError(error) {
        const errorInfo = ErrorHandler.classifyError(error);
        
        const errorUI = createErrorUI(
            errorInfo.type, 
            'данные проектов',
            () => {
                errorUI.remove();
                this._loadAndRenderData();
            },
            null
        );
        
        this.contentElement.appendChild(errorUI);
    }

    /**
     * Показывает модальное окно поиска
     */
    _showSearchModal(message) {
        const searchModal = document.createElement('div');
        searchModal.className = 'modal';
        searchModal.style.display = 'flex';
        
        searchModal.innerHTML = `
            <div class="modal-content" style="max-width: 500px;">
                <span class="close-modal">×</span>
                <h2>🔍 Поиск</h2>
                <p>${message}</p>
                <button class="gradient-button" onclick="this.closest('.modal').remove()">Понятно</button>
            </div>
        `;
        
        document.body.appendChild(searchModal);
        
        // Обработчики закрытия
        searchModal.querySelector('.close-modal').addEventListener('click', () => {
            searchModal.remove();
        });
        
        searchModal.addEventListener('click', (e) => {
            if (e.target === searchModal) {
                searchModal.remove();
            }
        });
    }
} 