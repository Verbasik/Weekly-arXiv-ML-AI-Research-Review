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
        // Поддержка новой пиксельной навигации (.nav-search) и старой разметки (.search-bar)
        this.searchInput = document.querySelector('.nav-search input') || document.querySelector('.search-bar input');
        this.searchButton = document.querySelector('.nav-search button') || document.querySelector('.search-bar button');
        this.searchSuggestions = null; // контейнер выпадающих подсказок
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

            // Инициализируем контейнер подсказок под строкой поиска
            this._ensureSearchSuggestionsContainer();

            // Вставляем служебные карточки (Contribute/Contact) после последнего года (например, 2024)
            try {
                const yearIds = years.map(y => y.year);
                const minYear = yearIds.reduce((a, b) => (a < b ? a : b), yearIds[0]);
                this._appendUtilityCardsToYear(minYear);
                this._scrollToUtilityAnchor();
            } catch (e) { /* non-fatal */ }

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
            // Сбой загрузки тегов не критичен
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
            // Сбой загрузки trending статей не критичен
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
        const q = (query || '').trim();
        if (q.length < 2) {
            this._clearSearchSuggestions();
            this.currentSearchQuery = '';
            this._applyCurrentFilter();
            return;
        }

        try {
            const results = await this.service.searchResearch(q);
            this._renderSearchSuggestions(results, q);
        } catch (error) {
            console.error('Search error:', error);
            this._showSearchModal('Не удалось выполнить поиск. Попробуйте позже.');
        }
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
                const value = e.target.value || '';
                if (!value.trim()) {
                    this.currentSearchQuery = '';
                    this._applyCurrentFilter();
                    this._clearSearchSuggestions();
                } else {
                    this._debouncedSuggest(value);
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

        // Закрытие подсказок по клику вне поля
        document.addEventListener('click', (e) => {
            const host = this.searchInput?.closest('.nav-search') || this.searchInput?.closest('.search-bar');
            if (!host) return;
            if (!host.contains(e.target)) {
                this._clearSearchSuggestions();
            }
        });

        // Закрытие подсказок по Escape
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this._clearSearchSuggestions();
            }
        });

        // Обработка unhandled promise rejections
        window.addEventListener('unhandledrejection', (event) => {
            console.error('Unhandled promise rejection:', event.reason);
            event.preventDefault();
            this._showErrorNotification('⚠️ Произошла неожиданная ошибка');
        });

        // Обработчик кастомного события открытия из WeekCard
        document.addEventListener('openReview', (event) => {
            const { year, weekId, title, useFullscreenModal } = event.detail;
            this.modal.open(year, weekId, title, useFullscreenModal);
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
     * Добавляет карточки Contribute/Contact в конец указанного года
     */
    _appendUtilityCardsToYear(year) {
        const section = document.getElementById(String(year));
        if (!section) return;
        const grid = section.querySelector('.weeks-grid');
        if (!grid) return;

        const contribute = document.createElement('div');
        contribute.className = 'pixel-card week-card';
        contribute.id = 'contribute';
        contribute.innerHTML = `
            <div class="pixel-flex pixel-flex-between pixel-mb-2" style="align-items: flex-start;">
                <div class="pixel-flex pixel-gap-2">
                    <div style="font-size: 2rem;">🤝</div>
                    <div>
                        <h3 class="week-card-title" style="font-family: var(--pixel-font-display); font-size: var(--pixel-font-base); margin-bottom: var(--px-unit-half); color: var(--pixel-ink);">Contribute</h3>
                        <div class="pixel-badge pixel-badge--success" data-icon="⭐">Community Quest</div>
                    </div>
                </div>
            </div>
            <p class="week-card-desc" style="font-size: var(--pixel-font-sm); color: var(--pixel-ink-soft);">
                Help improve TWRB: ideas, issues, pull requests — everything matters!
            </p>
            <div class="pixel-flex pixel-gap-2 pixel-mt-2">
                <a href="https://github.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review" target="_blank" class="pixel-btn pixel-btn--sm">🐙 Repo</a>
                <a href="https://github.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/issues/new/choose" target="_blank" class="pixel-btn pixel-btn--sm">📝 Issue</a>
                <a href="https://github.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/fork" target="_blank" class="pixel-btn pixel-btn--sm">🍴 Fork & PR</a>
                <a href="https://github.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review#readme" target="_blank" class="pixel-btn pixel-btn--sm">📖 Readme</a>
            </div>
        `;

        const contact = document.createElement('div');
        contact.className = 'pixel-card week-card';
        contact.id = 'contact';
        contact.innerHTML = `
            <div class="pixel-flex pixel-flex-between pixel-mb-2" style="align-items: flex-start;">
                <div class="pixel-flex pixel-gap-2">
                    <div style="font-size: 2rem;">✉️</div>
                    <div>
                        <h3 class="week-card-title" style="font-family: var(--pixel-font-display); font-size: var(--pixel-font-base); margin-bottom: var(--px-unit-half); color: var(--pixel-ink);">Contact</h3>
                        <div class="pixel-badge" data-icon="💬">Say Hello</div>
                    </div>
                </div>
            </div>
            <p class="week-card-desc" style="font-size: var(--pixel-font-sm); color: var(--pixel-ink-soft);">
                Questions or collaboration ideas? Reach out on your favorite channel.
            </p>
            <div class="pixel-flex pixel-gap-2 pixel-mt-2">
                <a href="mailto:verbasik2018@gmail.com" class="pixel-btn pixel-btn--sm">📬 Email</a>
                <a href="https://t.me/Verbasik" target="_blank" rel="noopener" class="pixel-btn pixel-btn--sm">📨 Telegram</a>
                <a href="https://www.linkedin.com/in/verbasik/" target="_blank" rel="noopener" class="pixel-btn pixel-btn--sm">💼 LinkedIn</a>
            </div>
        `;

        grid.appendChild(contribute);
        grid.appendChild(contact);
    }

    /**
     * Прокручивает к #contribute/#contact, если они в hash
     */
    _scrollToUtilityAnchor() {
        const id = (location.hash || '').replace('#','');
        if (!id) return;
        if (id === 'contribute' || id === 'contact') {
            const el = document.getElementById(id);
            if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }

    /**
     * Создает контейнер подсказок под полем поиска
     */
    _ensureSearchSuggestionsContainer() {
        if (!this.searchInput) return;
        const host = this.searchInput.closest('.nav-search') || this.searchInput.closest('.search-bar');
        if (!host) return;
        if (!this.searchSuggestions) {
            const container = document.createElement('div');
            container.className = 'search-suggestions pixel-card';
            container.setAttribute('role', 'listbox');
            container.style.display = 'none';
            // визуально под инпутом
            host.style.position = 'relative';
            host.appendChild(container);
            this.searchSuggestions = container;
        }
    }

    /**
     * Рендерит подсказки
     */
    _renderSearchSuggestions(results, query) {
        this._ensureSearchSuggestionsContainer();
        if (!this.searchSuggestions) return;

        if (!results || results.length === 0) {
            this.searchSuggestions.innerHTML = `<div class="search-suggestion empty">Ничего не найдено</div>`;
            this.searchSuggestions.style.display = 'block';
            return;
        }

        const max = 8;
        const html = results.slice(0, max).map(({ year, week }) => {
            const id = week.getId();
            const title = week.title;
            return `<button class="search-suggestion" role="option" data-year="${year}" data-id="${id}">${title}</button>`;
        }).join('');

        this.searchSuggestions.innerHTML = html;
        this.searchSuggestions.style.display = 'block';

        this.searchSuggestions.querySelectorAll('.search-suggestion').forEach(btn => {
            btn.addEventListener('click', () => {
                const year = btn.getAttribute('data-year');
                const id = btn.getAttribute('data-id');
                const title = btn.textContent;
                this.modal.open(year, id, title, true);
                this._clearSearchSuggestions();
            });
        });
    }

    _clearSearchSuggestions() {
        if (!this.searchSuggestions) return;
        this.searchSuggestions.innerHTML = '';
        this.searchSuggestions.style.display = 'none';
    }

    _debouncedSuggest(value) {
        clearTimeout(this._suggestTimer);
        this._suggestTimer = setTimeout(() => {
            const q = (value || '').trim();
            if (q.length >= 2) {
                this._performSearch(q);
            } else {
                this._clearSearchSuggestions();
            }
        }, 180);
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
        });
        
        window.addEventListener('offline', () => {
            indicator.className = 'network-status offline';
            indicator.innerHTML = '📡 Нет подключения';
            indicator.style.opacity = '1';
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
