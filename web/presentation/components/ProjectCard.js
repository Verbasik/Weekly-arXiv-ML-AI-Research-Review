/**
 * Project Card Component - Presentation Layer
 * Компонент карточки проекта для Agents секции
 */
export class ProjectCard {
    constructor(project, githubConfig, dataSource = null) {
        this.project = project;
        this.githubConfig = githubConfig;
        this.dataSource = dataSource;
        this.element = null;
    }

    /**
     * Создает DOM элемент карточки - УПРОЩЕННАЯ ВЕРСИЯ КАК В WEEKCARD
     */
    createElement() {
        const card = document.createElement('div');
        card.className = 'week-card project-card';
        card.setAttribute('data-project', this.project.getId());
        
        card.innerHTML = this._getCardHTML();
        this._attachEventListeners(card);
        this.element = card;
        
        return card;
    }

    /**
     * Генерирует HTML содержимое карточки - УПРОЩЕННАЯ ВЕРСИЯ
     */
    _getCardHTML() {
        const meta = this.project.getFormattedMeta();
        // НЕ ИСПОЛЬЗУЕМ СЛОЖНУЮ КОНФИГУРАЦИЮ - передаем null для fallback
        const resources = this.project.getFormattedResources(this.githubConfig.githubRepo, this.githubConfig.githubBranch, null);
        
        return `
            <div class="week-card-header">
                <h3 class="week-card-title">${this.project.title}</h3>
                <div class="week-card-meta">
                    ${meta.map(item => this._getMetaItemHTML(item)).join('')}
                </div>
            </div>
            <div class="week-card-body">
                <p class="week-card-desc">${this.project.description}</p>
                <button type="button" class="gradient-button read-review">Read Review</button>
            </div>
            <div class="week-card-footer">
                ${resources.map(resource => this._getResourceHTML(resource)).join('')}
            </div>
            <!-- Невидимый оверлей-ссылка для клика по всей карточке -->
            <a href="#agents/${this.project.getId()}" class="card-overlay-link" aria-label="Open ${this.project.title}"></a>
        `;
    }

    /**
     * Создает HTML для элемента метаданных
     */
    _getMetaItemHTML(item) {
        const typeClass = item.type ? `meta-${item.type}` : '';
        return `<span class="meta-item ${typeClass}"><i class="${item.icon}"></i> ${item.text}</span>`;
    }

    /**
     * Создает HTML для ресурса - УПРОЩЕННАЯ ВЕРСИЯ КАК НА ГЛАВНОЙ СТРАНИЦЕ
     */
    _getResourceHTML(resource) {
        if (!resource.url) {
            return `<span class="disabled"><i class="${resource.icon}"></i> ${resource.text}</span>`;
        }

        // ПРОСТАЯ ЛОГИКА КАК В WEEKCARD - ВСЕГДА ССЫЛКИ
        return `<a href="${resource.url}" target="_blank" rel="noopener noreferrer" class="resource-link">
            <i class="${resource.icon}"></i> ${resource.text}
        </a>`;
    }

    /**
     * Привязывает обработчики событий
     */
    _attachEventListeners(card) {
        // Обработчик для кнопки "Read Review"
        const readReviewButton = card.querySelector('.read-review');
        if (readReviewButton) {
            readReviewButton.addEventListener('click', (e) => {
                e.stopPropagation();
                this._onReadReview();
            });
        }

        // Обработчики для тегов технологий (если есть)
        const techTags = card.querySelectorAll('.tech-tag');
        techTags.forEach(tag => {
            tag.addEventListener('click', (e) => {
                e.stopPropagation();
                const technology = tag.textContent.trim();
                this._onTechClick(technology);
            });
        });

        // УПРОЩЕННАЯ ЛОГИКА - НЕ БЛОКИРУЕМ КЛИКИ ПО ССЫЛКАМ КАК НА ГЛАВНОЙ СТРАНИЦЕ
        // Обработчик для клика по карточке (открытие по клику на карточку)
        card.addEventListener('click', (e) => {
            // Проверяем, что клик не по ссылке или кнопке - КАК В WEEKCARD
            if (!e.target.closest('a') && !e.target.closest('button')) {
                this._onReadReview();
            }
        });

        // Добавляем курсор pointer для интерактивности
        card.style.cursor = 'pointer';
    }

    /**
     * Обработчик нажатия на "Read Review"
     */
    _onReadReview() {
        // Создаем и отправляем кастомное событие
        const event = new CustomEvent('projectCardClicked', {
            detail: {
                projectId: this.project.getId(),
                title: this.project.title
            },
            bubbles: true
        });
        
        document.dispatchEvent(event);
    }

    /**
     * Обработчик клика по технологии
     */
    _onTechClick(technology) {
        // Здесь можно добавить логику фильтрации по технологии
    }

    /**
     * Обновляет карточку новыми данными проекта
     */
    update(project) {
        this.project = project;
        if (this.element) {
            this.element.innerHTML = this._getCardHTML();
            this._attachEventListeners(this.element);
        }
    }

    /**
     * Проверяет соответствие поисковому запросу
     */
    matchesSearch(query) {
        const searchTerm = query.toLowerCase();
        return (
            this.project.title.toLowerCase().includes(searchTerm) ||
            this.project.description.toLowerCase().includes(searchTerm) ||
            this.project.tags.some(tag => tag.toLowerCase().includes(searchTerm)) ||
            this.project.technologies.some(tech => tech.toLowerCase().includes(searchTerm))
        );
    }

    /**
     * Проверяет наличие тега
     */
    hasTag(tag) {
        return this.project.hasTag(tag);
    }

    /**
     * Проверяет наличие технологии
     */
    hasTechnology(tech) {
        return this.project.technologies.some(technology => 
            technology.toLowerCase().includes(tech.toLowerCase())
        );
    }

    /**
     * Проверяет уровень сложности
     */
    hasDifficulty(difficulty) {
        return this.project.difficulty === difficulty;
    }

    /**
     * Показывает карточку
     */
    show() {
        if (this.element) {
            this.element.style.display = 'block';
            this.element.style.opacity = '1';
        }
    }

    /**
     * Скрывает карточку
     */
    hide() {
        if (this.element) {
            this.element.style.display = 'none';
            this.element.style.opacity = '0';
        }
    }

    /**
     * Подсвечивает поисковый термин
     */
    highlight(searchTerm) {
        if (!this.element || !searchTerm) return;

        const elements = this.element.querySelectorAll('.week-card-title, .week-card-desc');
        elements.forEach(element => {
            const originalText = element.textContent;
            const highlightedText = this._highlightText(originalText, searchTerm);
            if (highlightedText !== originalText) {
                element.innerHTML = highlightedText;
            }
        });
    }

    /**
     * Убирает подсветку
     */
    removeHighlight() {
        if (!this.element) return;

        const highlightedElements = this.element.querySelectorAll('.highlight');
        highlightedElements.forEach(element => {
            const parent = element.parentNode;
            parent.replaceChild(document.createTextNode(element.textContent), element);
            parent.normalize();
        });
    }

    /**
     * Подсвечивает текст
     */
    _highlightText(text, searchTerm) {
        const regex = new RegExp(`(${searchTerm})`, 'gi');
        return text.replace(regex, '<span class="highlight">$1</span>');
    }
} 