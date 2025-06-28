/**
 * Project Card Component - Presentation Layer
 * Компонент карточки проекта для Agents секции
 */
export class ProjectCard {
    constructor(project, githubConfig) {
        this.project = project;
        this.githubConfig = githubConfig;
        this.element = null;
    }

    /**
     * Создает DOM элемент карточки
     */
    createElement() {
        const card = document.createElement('div');
        card.className = 'week-card project-card';
        card.innerHTML = this._getCardHTML();
        
        this._attachEventListeners(card);
        this.element = card;
        
        return card;
    }

    /**
     * Генерирует HTML содержимое карточки
     */
    _getCardHTML() {
        const meta = this.project.getFormattedMeta();
        const resources = this.project.getFormattedResources(this.githubConfig.githubRepo, this.githubConfig.githubBranch);
        
        // Формируем HTML для метаданных
        const metaHTML = meta.map(item => {
            if (item.type === 'tag') {
                return `<span class="mono"><i class="${item.icon}"></i> ${item.text}</span>`;
            }
            return `<span><i class="${item.icon}"></i> ${item.text}</span>`;
        }).join('');

        // Формируем HTML для ресурсов в футере (как в WeekCard)
        const resourcesHTML = resources.map(resource => {
            if (resource.url) {
                return `<a href="${resource.url}" target="_blank" rel="noopener noreferrer"><i class="${resource.icon}"></i> ${resource.text}</a>`;
            }
            return `<span><i class="${resource.icon}"></i> ${resource.text}</span>`;
        }).join('');

        return `
            <div class="week-card-header">
                <h3 class="week-card-title">${this.project.title}</h3>
            </div>
            
            <div class="week-card-body">
                <div class="week-card-meta">${metaHTML}</div>
                <p class="week-card-desc">${this.project.getSummary()}</p>
                <button class="gradient-button read-review">Read Review</button>
            </div>
            
            <div class="week-card-footer">
                ${resourcesHTML}
            </div>
        `;
    }

    /**
     * Генерирует HTML для элемента метаданных
     */
    _getMetaItemHTML(item) {
        const className = item.type ? `meta-${item.type}` : '';
        return `<span class="meta-item ${className}"><i class="${item.icon}"></i> ${item.text}</span>`;
    }

    /**
     * Генерирует HTML для ресурса
     */
    _getResourceHTML(resource) {
        if (resource.url) {
            return `<a href="${resource.url}" target="_blank" rel="noopener noreferrer">
                <i class="${resource.icon}"></i> ${resource.text}
            </a>`;
        } else {
            return `<span><i class="${resource.icon}"></i> ${resource.text}</span>`;
        }
    }

    /**
     * Добавляет обработчики событий
     */
    _attachEventListeners(card) {
        const readButton = card.querySelector('.read-review');
        if (readButton) {
            readButton.addEventListener('click', (e) => {
                e.preventDefault();
                this._onReadReview();
            });
        }

        // Обработчик для клика по карточке (открытие по клику на карточку)
        card.addEventListener('click', (e) => {
            // Проверяем, что клик не по ссылке или кнопке
            if (!e.target.closest('a') && !e.target.closest('button')) {
                this._onReadReview();
            }
        });

        // Добавляем курсор pointer для интерактивности
        card.style.cursor = 'pointer';
    }

    /**
     * Обработчик клика на "Read Review"
     */
    _onReadReview() {
        // Отправляем custom event для открытия модального окна (как в WeekCard)
        const event = new CustomEvent('projectCardClicked', {
            detail: {
                projectId: this.project.getId(),
                title: this.project.title
            }
        });
        
        document.dispatchEvent(event);
        console.log(`Opening project: ${this.project.getId()}`);
    }

    /**
     * Обработчик клика на технологию
     */
    _onTechClick(technology) {
        const event = new CustomEvent('technologyFilter', {
            detail: { technology }
        });
        
        document.dispatchEvent(event);
        console.log(`Filter by technology: ${technology}`);
    }

    /**
     * Обновляет данные карточки
     */
    update(project) {
        this.project = project;
        if (this.element) {
            this.element.innerHTML = this._getCardHTML();
            this._attachEventListeners(this.element);
        }
    }

    /**
     * Проверяет, соответствует ли карточка поисковому запросу
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
     * Проверяет, есть ли у карточки указанный тег
     */
    hasTag(tag) {
        return this.project.hasTag(tag);
    }

    /**
     * Проверяет технологию
     */
    hasTechnology(tech) {
        return this.project.technologies.some(technology => 
            technology.toLowerCase().includes(tech.toLowerCase())
        );
    }

    /**
     * Проверяет сложность
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
            this.element.classList.remove('hidden');
        }
    }

    /**
     * Скрывает карточку
     */
    hide() {
        if (this.element) {
            this.element.style.display = 'none';
            this.element.classList.add('hidden');
        }
    }

    /**
     * Подсвечивает карточку при поиске
     */
    highlight(searchTerm) {
        if (!this.element || !searchTerm) return;

        const title = this.element.querySelector('.week-card-title');
        const description = this.element.querySelector('.week-card-desc');
        
        if (title) {
            title.innerHTML = this._highlightText(this.project.title, searchTerm);
        }
        
        if (description) {
            description.innerHTML = this._highlightText(this.project.getSummary(), searchTerm);
        }
    }

    /**
     * Убирает подсветку
     */
    removeHighlight() {
        if (!this.element) return;

        const title = this.element.querySelector('.week-card-title');
        const description = this.element.querySelector('.week-card-desc');
        
        if (title) {
            title.textContent = this.project.title;
        }
        
        if (description) {
            description.textContent = this.project.getSummary();
        }
    }

    /**
     * Подсвечивает текст поискового запроса
     */
    _highlightText(text, searchTerm) {
        if (!searchTerm) return text;
        
        const regex = new RegExp(`(${searchTerm})`, 'gi');
        return text.replace(regex, '<mark>$1</mark>');
    }
} 