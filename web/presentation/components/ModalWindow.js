import { ErrorHandler, createErrorUI } from '../../infrastructure/error/ErrorHandler.js';

/**
 * Modal Window Component - Presentation Layer
 * Универсальный компонент модального окна для отображения markdown контента
 */
export class ModalWindow {
    constructor(modalElement, service) {
        this.modal = modalElement;
        this.service = service; // Может быть ResearchService или AgentsService
        this.markdownContent = modalElement?.querySelector('#markdown-content') || modalElement?.querySelector('.markdown-body');
        this.loader = modalElement?.querySelector('.loader');
        
        // Определяем тип сервиса
        this.serviceType = this._detectServiceType(service);
        
        this._initializeEventListeners();
    }

    /**
     * Определяет тип сервиса
     */
    _detectServiceType(service) {
        if (service && typeof service.getWeekMarkdown === 'function') {
            return 'research';
        } else if (service && typeof service.getProjectMarkdown === 'function') {
            return 'agents';
        }
        return 'research';
    }

    /**
     * Инициализирует обработчики событий
     */
    _initializeEventListeners() {
        if (!this.modal) return;

        // Закрытие по клику на X
        const closeButton = this.modal.querySelector('.close-modal');
        if (closeButton) {
            closeButton.addEventListener('click', () => {
                this.close();
            });
        }

        // Закрытие по клику на фон
        this.modal.addEventListener('click', (e) => {
            if (e.target === this.modal) {
                this.close();
            }
        });

        // Закрытие по Escape
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.isOpen()) {
                this.close();
            }
        });
    }

    /**
     * Открывает модальное окно
     * Универсальный метод для research и agents
     */
    async open(year, weekId, title) {
        if (!this.modal || !this.markdownContent) return;

        // Устанавливаем заголовок
        this._setTitle(title);

        // Показываем модальное окно
        this.modal.style.display = 'flex';
        document.body.style.overflow = 'hidden';

        // Загружаем контент в зависимости от типа сервиса
        const success = await this._loadMarkdown(year, weekId);

        if (success) {
            // Обновляем URL только при успехе
            this._updateUrl(year, weekId);
        }
    }

    /**
     * Закрывает модальное окно
     */
    close() {
        if (!this.modal) return;

        this.modal.style.display = 'none';
        document.body.style.overflow = '';
        
        if (this.markdownContent) {
            this.markdownContent.innerHTML = '';
        }
        
        // Сбрасываем URL
        this._resetUrl();
    }

    /**
     * Проверяет, открыто ли модальное окно
     */
    isOpen() {
        return this.modal && this.modal.style.display === 'flex';
    }

    /**
     * Загружает markdown контент
     */
    async _loadMarkdown(year, weekId) {
        if (!this.markdownContent || !this.loader) {
            console.error("Markdown content area or loader not found.");
            return false;
        }

        // Показываем индикатор загрузки
        this.loader.style.display = 'block';
        
        // Разные сообщения для разных типов контента
        const loadingMessage = this.serviceType === 'agents' 
            ? `Загрузка проекта "${year}"...`
            : `Загрузка статьи "${year}/${weekId}"...`;
            
        this.markdownContent.innerHTML = `
            <div class="loading-content">
                <div class="loader"></div>
                <p>${loadingMessage}</p>
                <p class="loading-tip">💡 Обычно это занимает несколько секунд</p>
            </div>
        `;

        try {
            let markdown;
            
            // Получаем markdown в зависимости от типа сервиса
            if (this.serviceType === 'agents') {
                // Для агентов year содержит projectId
                markdown = await this.service.getProjectMarkdown(year);
            } else {
                // Для исследований
                markdown = await this.service.getWeekMarkdown(year, weekId);
            }
            
            // Обрабатываем markdown
            const html = await this._processMarkdown(markdown);
            
            // Отображаем контент
            this.markdownContent.innerHTML = html;
            
            // Рендерим MathJax если доступен
            await this._renderMathJax();
            
            this.loader.style.display = 'none';
            return true;

        } catch (error) {
            console.error('Error loading markdown:', error);
            
            // Определяем тип ошибки
            const errorInfo = ErrorHandler.classifyError(error);
            
            // Разные сообщения ошибок для разных типов контента
            const errorContext = this.serviceType === 'agents' 
                ? `проект "${year}"`
                : `статья "${year}/${weekId}"`;
            
            // Создаем улучшенный error UI
            const errorUI = createErrorUI(
                errorInfo.type,
                errorContext,
                () => {
                    // Retry callback
                    this._loadMarkdown(year, weekId);
                },
                () => {
                    // Back callback - закрываем модальное окно
                    this.close();
                }
            );
            
            this.markdownContent.innerHTML = '';
            this.markdownContent.appendChild(errorUI);
            this.loader.style.display = 'none';
            
            // Автоматическая попытка перезагрузки при восстановлении соединения
            if (errorInfo.type === 'offline') {
                const handleOnline = () => {
                    this._loadMarkdown(year, weekId);
                    window.removeEventListener('online', handleOnline);
                };
                window.addEventListener('online', handleOnline);
            }
            
            return false;
        }
    }

    /**
     * Обрабатывает markdown в HTML
     */
    async _processMarkdown(markdown) {
        // 1. Изоляция формул MathJax
        const mathPlaceholders = {};
        let placeholderId = 0;
        const mathRegex = /(\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]|\$(?:[^$\\]|\\.)*?\$|\\\((?:[^)\\]|\\.)*?\\\))/g;
        
        const processedMarkdown = markdown.replace(mathRegex, (match) => {
            const id = `mathjax-placeholder-${placeholderId++}`;
            mathPlaceholders[id] = match;
            return `<span id="${id}" style="display: none;"></span>`;
        });

        // 2. Преобразование Markdown в HTML
        if (typeof marked === 'undefined') {
            throw new Error("Библиотека Marked.js не загружена. Попробуйте обновить страницу.");
        }
        
        let html;
        try {
            html = marked.parse(processedMarkdown);
        } catch (markdownError) {
            throw new Error(`Ошибка обработки Markdown: ${markdownError.message}`);
        }

        // 3. Создаем временный элемент для работы с DOM
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = html;

        // 4. Восстановление формул
        Object.keys(mathPlaceholders).forEach(id => {
            const placeholderElement = tempDiv.querySelector(`#${id}`);
            if (placeholderElement) {
                placeholderElement.replaceWith(document.createTextNode(mathPlaceholders[id]));
            }
        });

        return tempDiv.innerHTML;
    }

    /**
     * Рендерит MathJax формулы
     */
    async _renderMathJax() {
        try {
            if (typeof MathJax !== 'undefined' && MathJax.typesetPromise) {
                MathJax.texReset?.();
                MathJax.typesetClear?.([this.markdownContent]);
                await MathJax.typesetPromise([this.markdownContent]);
            } else {
                console.warn("MathJax 3 not found or not configured.");
            }
        } catch (mathJaxError) {
            console.warn("MathJax rendering failed:", mathJaxError);
            // Не выбрасываем ошибку, так как статья может отображаться без формул
        }
    }

    /**
     * Устанавливает заголовок модального окна
     */
    _setTitle(title) {
        const modalContentDiv = this.modal.querySelector('.modal-content');
        let titleElement = modalContentDiv?.querySelector('h2.modal-title');
        
        if (!titleElement) {
            titleElement = document.createElement('h2');
            titleElement.className = 'modal-title';
            titleElement.style.marginTop = '0';
            titleElement.style.marginBottom = '1rem';
            modalContentDiv?.insertBefore(titleElement, this.markdownContent);
        }
        
        titleElement.textContent = title;
    }

    /**
     * Обновляет URL с хешем
     */
    _updateUrl(year, weekId) {
        if (this.serviceType === 'agents') {
            // Для агентов используем только projectId
            window.location.hash = `#agents/${year}`;
        } else {
            // Для исследований используем year/weekId
            window.location.hash = `#${year}/${weekId}`;
        }
    }

    /**
     * Сбрасывает URL
     */
    _resetUrl() {
        history.replaceState(null, null, ' ');
    }

    /**
     * Проверяет URL hash и открывает соответствующий контент
     */
    checkUrlHash() {
        const hash = window.location.hash.substring(1); // Убираем #
        if (!hash) return;

        if (hash.startsWith('agents/')) {
            // Обработка URL для агентов: #agents/projectId
            const projectId = hash.substring(7); // Убираем 'agents/'
            if (projectId && this.serviceType === 'agents') {
                // Нужно получить title проекта из сервиса
                this._openProjectFromHash(projectId);
            }
        } else if (hash.includes('/')) {
            // Обработка URL для исследований: #year/weekId
            const [year, weekId] = hash.split('/');
            if (year && weekId && this.serviceType === 'research') {
                this.open(year, weekId, `${year} / ${weekId}`);
            }
        }
    }

    /**
     * Открывает проект из hash URL
     */
    async _openProjectFromHash(projectId) {
        try {
            const project = await this.service.getProjectData(projectId);
            if (project) {
                this.open(projectId, projectId, project.title);
            }
        } catch (error) {
            console.error('Error opening project from hash:', error);
        }
    }
} 