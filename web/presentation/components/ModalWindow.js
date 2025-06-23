import { ErrorHandler, createErrorUI } from '../../infrastructure/error/ErrorHandler.js';

/**
 * Modal Window Component - Presentation Layer
 * Компонент для отображения модального окна с markdown контентом
 */
export class ModalWindow {
    constructor(modalElement, researchService) {
        this.modal = modalElement;
        this.researchService = researchService;
        this.markdownContent = modalElement.querySelector('#markdown-content');
        this.loader = modalElement.querySelector('.loader');
        this.closeButton = modalElement.querySelector('.close-modal');
        
        this._initializeEventListeners();
    }

    /**
     * Инициализирует обработчики событий
     */
    _initializeEventListeners() {
        // Закрытие по кнопке
        if (this.closeButton) {
            this.closeButton.addEventListener('click', () => this.close());
        }

        // Закрытие по клику вне модального окна
        this.modal.addEventListener('click', (event) => {
            if (event.target === this.modal) {
                this.close();
            }
        });

        // Закрытие по Escape
        document.addEventListener('keydown', (event) => {
            if (event.key === 'Escape' && this.isOpen()) {
                this.close();
            }
        });

        // Обработчик кастомного события открытия
        document.addEventListener('openReview', (event) => {
            const { year, weekId, title } = event.detail;
            this.open(year, weekId, title);
        });
    }

    /**
     * Открывает модальное окно
     */
    async open(year, weekId, title) {
        if (!this.modal || !this.markdownContent) return;

        // Устанавливаем заголовок
        this._setTitle(title);

        // Показываем модальное окно
        this.modal.style.display = 'flex';
        document.body.style.overflow = 'hidden';

        // Загружаем контент
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
        this.markdownContent.innerHTML = `
            <div class="loading-content">
                <div class="loader"></div>
                <p>Загрузка статьи "${year}/${weekId}"...</p>
                <p class="loading-tip">💡 Обычно это занимает несколько секунд</p>
            </div>
        `;

        try {
            // Получаем markdown через сервис
            const markdown = await this.researchService.getWeekMarkdown(year, weekId);
            
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
            
            // Создаем улучшенный error UI
            const errorUI = createErrorUI(
                errorInfo.type,
                `статья "${year}/${weekId}"`,
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
        window.location.hash = `#${year}/${weekId}`;
    }

    /**
     * Сбрасывает URL
     */
    _resetUrl() {
        history.pushState("", document.title, window.location.pathname + window.location.search);
    }

    /**
     * Проверяет URL хеш и открывает модальное окно если нужно
     */
    checkUrlHash() {
        const hash = window.location.hash;
        if (hash && hash.startsWith('#') && hash.includes('/')) {
            const parts = hash.substring(1).split('/');
            if (parts.length === 2 && parts[0] && parts[1]) {
                const year = parts[0];
                const weekId = parts[1];

                // Проверяем, не открыто ли уже это модальное окно
                const currentModalTitle = this.modal?.querySelector('.modal-content h2.modal-title');
                const expectedTitle = `Review ${year}/${weekId}`;
                
                if (!this.isOpen() || !currentModalTitle || !currentModalTitle.textContent.includes(`${year}/${weekId}`)) {
                    // Пытаемся найти заголовок из карточки
                    const card = document.querySelector(`.week-card[data-year="${year}"][data-week="${weekId}"]`);
                    const title = card?.querySelector('.week-card-title')?.textContent || expectedTitle;
                    
                    this.open(year, weekId, title);
                }
            } else {
                // Хеш не соответствует формату, закрываем окно
                if (this.isOpen()) {
                    this.close();
                }
            }
        } else {
            // Хеш пуст или не содержит '/', закрываем окно
            if (this.isOpen()) {
                this.close();
            }
        }
    }
} 