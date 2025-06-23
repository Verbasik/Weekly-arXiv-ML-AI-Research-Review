/**
 * Week Card Component - Presentation Layer
 * Компонент для отображения карточки недели
 */
export class WeekCard {
    constructor(year, week, githubConfig) {
        this.year = year;
        this.week = week;
        this.githubConfig = githubConfig;
    }

    /**
     * Создает DOM элемент карточки
     */
    createElement() {
        const card = document.createElement('div');
        card.className = 'week-card';
        card.setAttribute('data-week', this.week.getId());
        card.setAttribute('data-year', this.year);

        card.innerHTML = this._getCardHTML();
        this._attachEventListeners(card);
        
        return card;
    }

    /**
     * Генерирует HTML разметку карточки
     */
    _getCardHTML() {
        const meta = this.week.getFormattedMeta();
        const resources = this.week.getFormattedResources(
            this.githubConfig.githubRepo, 
            this.githubConfig.githubBranch
        );

        const metaHTML = meta.map(item => {
            if (item.type === 'tag') {
                return `<span class="mono"><i class="${item.icon}"></i> ${item.text}</span>`;
            }
            return `<span><i class="${item.icon}"></i> ${item.text}</span>`;
        }).join('');

        const resourcesHTML = resources.map(resource => {
            if (resource.url) {
                return `<a href="${resource.url}" target="_blank"><i class="${resource.icon}"></i> ${resource.text}</a>`;
            }
            return `<span><i class="${resource.icon}"></i> ${resource.text}</span>`;
        }).join('');

        return `
            <div class="week-card-header">
                <h3 class="week-card-title">${this.week.title}</h3>
            </div>
            <div class="week-card-body">
                <div class="week-card-meta">${metaHTML}</div>
                <p class="week-card-desc">${this.week.getSummary()}</p>
                <button class="gradient-button read-review">Read Review</button>
            </div>
            <div class="week-card-footer">
                ${resourcesHTML}
            </div>
        `;
    }

    /**
     * Прикрепляет обработчики событий
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
     * Обработчик открытия обзора
     */
    _onReadReview() {
        // Создаем кастомное событие для открытия модального окна
        const event = new CustomEvent('openReview', {
            detail: {
                year: this.year,
                weekId: this.week.getId(),
                title: this.week.title
            }
        });
        
        document.dispatchEvent(event);
    }

    /**
     * Обновляет содержимое карточки
     */
    update(week) {
        this.week = week;
        // Перерисовываем карточку с новыми данными
        // Это может быть полезно для динамических обновлений
    }

    /**
     * Проверяет, соответствует ли карточка поисковому запросу
     */
    matchesSearch(query) {
        const searchTerm = query.toLowerCase();
        return (
            this.week.title.toLowerCase().includes(searchTerm) ||
            this.week.description.toLowerCase().includes(searchTerm) ||
            this.week.tags.some(tag => tag.toLowerCase().includes(searchTerm))
        );
    }

    /**
     * Проверяет, содержит ли карточка указанный тег
     */
    hasTag(tag) {
        return this.week.hasTag(tag);
    }

    /**
     * Показывает карточку
     */
    show() {
        const element = document.querySelector(`[data-week="${this.week.getId()}"][data-year="${this.year}"]`);
        if (element) {
            element.style.display = 'block';
        }
    }

    /**
     * Скрывает карточку
     */
    hide() {
        const element = document.querySelector(`[data-week="${this.week.getId()}"][data-year="${this.year}"]`);
        if (element) {
            element.style.display = 'none';
        }
    }

    /**
     * Подсвечивает карточку (например, при поиске)
     */
    highlight(searchTerm) {
        const element = document.querySelector(`[data-week="${this.week.getId()}"][data-year="${this.year}"]`);
        if (!element || !searchTerm) return;

        // Подсвечиваем текст в заголовке и описании
        const title = element.querySelector('.week-card-title');
        const description = element.querySelector('.week-card-desc');
        
        if (title) {
            title.innerHTML = this._highlightText(this.week.title, searchTerm);
        }
        
        if (description) {
            description.innerHTML = this._highlightText(this.week.getSummary(), searchTerm);
        }
    }

    /**
     * Убирает подсветку
     */
    removeHighlight() {
        const element = document.querySelector(`[data-week="${this.week.getId()}"][data-year="${this.year}"]`);
        if (!element) return;

        const title = element.querySelector('.week-card-title');
        const description = element.querySelector('.week-card-desc');
        
        if (title) {
            title.textContent = this.week.title;
        }
        
        if (description) {
            description.textContent = this.week.getSummary();
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