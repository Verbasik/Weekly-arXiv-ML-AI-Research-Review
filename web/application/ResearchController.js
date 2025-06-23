import { GitHubDataSource } from '../infrastructure/data/GitHubDataSource.js';
import { ResearchRepository } from '../domain/research/repositories/ResearchRepository.js';
import { ResearchService } from '../domain/research/services/ResearchService.js';
import { WeekCard } from '../presentation/components/WeekCard.js';
import { ModalWindow } from '../presentation/components/ModalWindow.js';
import { ErrorHandler, createErrorUI } from '../infrastructure/error/ErrorHandler.js';

/**
 * Research Controller - Application Layer
 * Координирует взаимодействие между всеми слоями приложения
 */
export class ResearchController {
    constructor(config) {
        this.config = config;
        this.githubConfig = {
            githubRepo: config.githubRepo,
            githubBranch: config.githubBranch
        };
        
        // Инициализируем слои архитектуры
        this.dataSource = new GitHubDataSource(this.githubConfig);
        this.repository = new ResearchRepository(this.dataSource);
        this.service = new ResearchService(this.repository);
        
        // DOM элементы
        this.contentElement = document.querySelector('.content');
        this.modalElement = document.getElementById('markdown-modal');
        this.searchInput = document.querySelector('.search-bar input');
        this.searchButton = document.querySelector('.search-bar button');
        this.backToTopButton = document.getElementById('back-to-top');
        
        // Компоненты презентации
        this.modal = new ModalWindow(this.modalElement, this.service);
        this.weekCards = new Map(); // Хранилище карточек недель
        
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
            // Инициализируем мониторинг сети
            this._initNetworkMonitoring();
            
            // Загружаем данные и рендерим интерфейс
            await this._loadAndRenderData();
            
            // Проверяем URL hash
            this.modal.checkUrlHash();
            
            console.log('Research application initialized successfully');
            
        } catch (error) {
            console.error('Failed to initialize research application:', error);
            this._showCriticalError(error);
        }
    }

    /**
     * Загружает и рендерит данные исследований
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
            const years = await this.service.getAllResearchData();

            // Удаляем индикатор загрузки
            loadingIndicator.remove();

            // Очищаем старые секции
            this._clearOldSections();

            // Создаем секции и карточки
            years.forEach(year => {
                const yearSection = this._createYearSection(year.year);
                year.getWeeks().forEach(week => {
                    this._createAndAddWeekCard(yearSection, year.year, week);
                });
            });

            // Обновляем фильтры в боковой панели
            this._updateSidebarFilters(years);

        } catch (error) {
            console.error('Error loading research data:', error);
            
            // Удаляем индикатор загрузки
            loadingIndicator.remove();
            
            // Показываем ошибку
            this._showDataLoadError(error);
        }
    }

    /**
     * Создает и добавляет карточку недели
     */
    _createAndAddWeekCard(yearSection, year, week) {
        const weekCard = new WeekCard(year, week, this.githubConfig);
        const cardElement = weekCard.createElement();
        
        const weeksGrid = yearSection.querySelector('.weeks-grid');
        if (weeksGrid) {
            weeksGrid.appendChild(cardElement);
            
            // Сохраняем ссылку на карточку для управления
            const cardKey = `${year}-${week.getId()}`;
            this.weekCards.set(cardKey, weekCard);
        }
    }

    /**
     * Создает секцию года
     */
    _createYearSection(year) {
        const yearSection = document.createElement('section');
        yearSection.id = year;
        yearSection.className = 'year-section';
        yearSection.innerHTML = `
            <h2 class="year-title section-heading">${year} Papers</h2>
            <div class="weeks-grid"></div>
        `;
        this.contentElement.appendChild(yearSection);
        return yearSection;
    }

    /**
     * Обновляет фильтры в боковой панели
     */
    _updateSidebarFilters(years) {
        // Обновляем фильтр по годам
        this._updateYearFilters(years.map(y => y.year));
        
        // Обновляем популярные теги
        this._updatePopularTags(years);
        
        // Обновляем trending papers
        this._updateTrendingPapers();
    }

    /**
     * Обновляет фильтр по годам
     */
    _updateYearFilters(years) {
        const yearFilterList = document.querySelector('.sidebar ul:first-of-type');
        if (!yearFilterList) return;

        yearFilterList.innerHTML = '';
        years.sort((a, b) => b - a).forEach((year, index) => {
            const li = document.createElement('li');
            const a = document.createElement('a');
            a.href = `#${year}`;
            a.textContent = year;
            if (index === 0) a.className = 'active';
            li.appendChild(a);
            yearFilterList.appendChild(li);

            // Обработчик клика для фильтра
            a.addEventListener('click', (e) => {
                yearFilterList.querySelectorAll('a').forEach(link => link.classList.remove('active'));
                a.classList.add('active');
                this._filterByYear(year);
            });
        });
    }

    /**
     * Обновляет популярные теги
     */
    async _updatePopularTags(years) {
        try {
            const popularTags = await this.service.getPopularTags(8);
            const tagsContainer = document.querySelector('.tags-container');
            
            if (tagsContainer) {
                tagsContainer.innerHTML = '';
                popularTags.forEach(({ tag, count }) => {
                    const tagElement = document.createElement('span');
                    tagElement.className = 'tag';
                    tagElement.textContent = tag;
                    tagElement.title = `${count} статей`;
                    
                    tagElement.addEventListener('click', () => {
                        this._filterByTag(tag);
                    });
                    
                    tagsContainer.appendChild(tagElement);
                });
            }
        } catch (error) {
            console.warn('Failed to load popular tags:', error);
        }
    }

    /**
     * Обновляет trending papers
     */
    async _updateTrendingPapers() {
        try {
            const trendingPapers = await this.service.getTrendingPapers(4);
            const trendingList = document.querySelector('.sidebar-section:last-child ul');
            
            if (trendingList) {
                trendingList.innerHTML = '';
                trendingPapers.forEach(({ year, week }) => {
                    const li = document.createElement('li');
                    const a = document.createElement('a');
                    a.href = `#${year}/${week.getId()}`;
                    a.textContent = week.title;
                    
                    a.addEventListener('click', (e) => {
                        e.preventDefault();
                        this.modal.open(year, week.getId(), week.title);
                    });
                    
                    li.appendChild(a);
                    trendingList.appendChild(li);
                });
            }
        } catch (error) {
            console.warn('Failed to load trending papers:', error);
        }
    }

    /**
     * Фильтрует карточки по году
     */
    _filterByYear(year) {
        this.currentFilter = { type: 'year', value: year };
        this._applyCurrentFilter();
        
        // Плавная прокрутка к секции
        const targetElement = document.getElementById(year.toString());
        if (targetElement) {
            targetElement.scrollIntoView({ behavior: 'smooth' });
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
        this.weekCards.forEach((weekCard, key) => {
            let shouldShow = true;
            
            if (this.currentFilter) {
                switch (this.currentFilter.type) {
                    case 'year':
                        const [cardYear] = key.split('-');
                        shouldShow = cardYear === this.currentFilter.value.toString();
                        break;
                    case 'tag':
                        shouldShow = weekCard.hasTag(this.currentFilter.value);
                        break;
                }
            }
            
            // Применяем поиск если есть
            if (shouldShow && this.currentSearchQuery) {
                shouldShow = weekCard.matchesSearch(this.currentSearchQuery);
            }
            
            if (shouldShow) {
                weekCard.show();
                if (this.currentSearchQuery) {
                    weekCard.highlight(this.currentSearchQuery);
                } else {
                    weekCard.removeHighlight();
                }
            } else {
                weekCard.hide();
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

        console.log(`Searching for: ${query}`);
        // Здесь будет реализована логика поиска
    }

    /**
     * Сбрасывает фильтры и поиск
     */
    _resetFilters() {
        this.currentFilter = null;
        this.currentSearchQuery = '';
        
        // Показываем все карточки и убираем подсветку
        this.weekCards.forEach(weekCard => {
            weekCard.show();
            weekCard.removeHighlight();
        });
        
        // Сбрасываем поисковый запрос
        if (this.searchInput) {
            this.searchInput.value = '';
        }
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
            
            // Сброс поиска при очистке поля
            this.searchInput.addEventListener('input', (e) => {
                if (!e.target.value.trim()) {
                    this.currentSearchQuery = '';
                    this._applyCurrentFilter();
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

        // URL hash изменения
        window.addEventListener('hashchange', () => {
            this.modal.checkUrlHash();
        });

        // Обработка unhandled promise rejections
        window.addEventListener('unhandledrejection', (event) => {
            console.error('Unhandled promise rejection:', event.reason);
            event.preventDefault();
            this._showErrorNotification('⚠️ Произошла неожиданная ошибка');
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
            <p>Загрузка статей...</p>
        `;
        return loadingIndicator;
    }

    /**
     * Очищает старые секции
     */
    _clearOldSections() {
        this.contentElement.querySelectorAll('.year-section:not(#home)').forEach(section => section.remove());
        this.weekCards.clear();
    }

    /**
     * Показывает критическую ошибку
     */
    _showCriticalError(error) {
        if (!this.contentElement) return;
        
        const criticalError = createErrorUI(
            'unknown',
            'приложение',
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
            'данные статей',
            () => {
                errorUI.remove();
                this._loadAndRenderData();
            },
            null
        );
        
        this.contentElement.appendChild(errorUI);
        
        // Автоматическая попытка перезагрузки при восстановлении соединения
        if (errorInfo.type === 'offline') {
            const handleOnline = () => {
                errorUI.remove();
                this._loadAndRenderData();
                window.removeEventListener('online', handleOnline);
            };
            window.addEventListener('online', handleOnline);
        }
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
                <div style="margin-top: 20px;">
                    <p><strong>Пока что вы можете:</strong></p>
                    <ul style="text-align: left; margin-top: 10px;">
                        <li>Просматривать статьи по годам в боковой панели</li>
                        <li>Использовать теги для фильтрации</li>
                        <li>Просматривать популярные статьи</li>
                    </ul>
                </div>
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

    /**
     * Показывает уведомление об ошибке
     */
    _showErrorNotification(message) {
        const notification = document.createElement('div');
        notification.className = 'error-notification';
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(244, 67, 54, 0.9);
            color: white;
            padding: 12px 16px;
            border-radius: 8px;
            font-size: 14px;
            z-index: 10000;
            max-width: 300px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        `;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.remove();
        }, 5000);
    }

    /**
     * Инициализирует мониторинг сети
     */
    _initNetworkMonitoring() {
        const indicator = this._createNetworkStatusIndicator();
        
        window.addEventListener('online', () => {
            indicator.className = 'network-status online';
            indicator.innerHTML = '🌐 Подключение восстановлено';
            
            setTimeout(() => {
                indicator.style.opacity = '0';
            }, 3000);
            
            console.log('Network connection restored');
        });
        
        window.addEventListener('offline', () => {
            indicator.className = 'network-status offline';
            indicator.innerHTML = '📡 Нет подключения';
            indicator.style.opacity = '1';
            
            console.log('Network connection lost');
        });
        
        // Проверяем начальное состояние
        if (!navigator.onLine) {
            indicator.className = 'network-status offline';
            indicator.innerHTML = '📡 Нет подключения';
            indicator.style.opacity = '1';
        }
    }

    /**
     * Создает индикатор статуса сети
     */
    _createNetworkStatusIndicator() {
        const indicator = document.createElement('div');
        indicator.className = 'network-status online';
        indicator.innerHTML = '🌐 Онлайн';
        document.body.appendChild(indicator);
        return indicator;
    }
} 