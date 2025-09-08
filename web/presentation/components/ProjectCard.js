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
        card.className = 'pixel-card project-card';
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
        const resources = this.project.getFormattedResources(
            this.githubConfig.githubRepo,
            this.githubConfig.githubBranch,
            null
        );

        // Конвертируем FA-иконки в эмодзи для пиксельной темы
        const iconMap = {
            'fas fa-calendar-alt': '📅',
            'fas fa-clock': '⏰',
            'fas fa-tag': '🏷️',
            'fas fa-fire': '🔥',
            'fas fa-star': '⭐',
            'fas fa-brain': '🧠',
            'fas fa-robot': '🤖',
            'fas fa-chart-line': '📊',
            'fas fa-file-pdf': '📄',
            'fas fa-code': '💻',
            'fas fa-play': '▶️',
            'fas fa-download': '⬇️',
            'fas fa-external-link-alt': '🔗'
        };

        const metaHTML = meta.map(item => {
            const emoji = iconMap[item.icon] || '📌';
            if (item.type === 'tag') {
                return `<span class="pixel-tag">${emoji} ${item.text}</span>`;
            }
            return `<span style="font-family: var(--pixel-font-display); font-size: var(--pixel-font-xs); color: var(--pixel-ink-soft);">${emoji} ${item.text}</span>`;
        }).join('');

        const resourcesHTML = resources.map(resource => {
            const emoji = iconMap[resource.icon] || '🔗';
            if (resource.url) {
                return `<a href="${resource.url}" target="_blank" rel="noopener" class="pixel-btn pixel-btn--sm" style="font-size: var(--pixel-font-xs);">${emoji} ${resource.text}</a>`;
            }
            return `<span class="pixel-badge" data-icon="${emoji}">${resource.text}</span>`;
        }).join('');

        // Случайная “сложность” как бейдж (для игрового ощущения)
        const difficulties = [
            { level: 'Beginner', emoji: '🟢', color: 'success' },
            { level: 'Intermediate', emoji: '🟡', color: 'warning' },
            { level: 'Advanced', emoji: '🟠', color: 'danger' },
            { level: 'Expert', emoji: '🔴', color: 'secondary' }
        ];
        const difficulty = difficulties[Math.floor(Math.random() * difficulties.length)];

        return `
            <!-- Game Cartridge Header -->
            <div class="pixel-flex pixel-flex-between pixel-mb-2" style="align-items: flex-start;">
                <div class="pixel-flex pixel-gap-2">
                    <div style="font-size: 2rem;">🤖</div>
                    <div>
                        <h3 class="project-card-title" style="font-family: var(--pixel-font-display); font-size: var(--pixel-font-base); margin-bottom: var(--px-unit-half); color: var(--pixel-ink);">
                            ${this.project.title}
                        </h3>
                        <div class="pixel-badge pixel-badge--${difficulty.color}" data-icon="${difficulty.emoji}">
                            ${difficulty.level} Quest
                        </div>
                    </div>
                </div>
                <div style="font-size: 1.5rem;">🧩</div>
            </div>

            <!-- Meta and Description -->
            <div class="pixel-mb-3">
                <div class="pixel-flex pixel-flex-wrap pixel-gap-1 pixel-mb-2">
                    ${metaHTML}
                </div>
                <p class="project-card-desc" style="font-size: var(--pixel-font-sm); line-height: var(--pixel-line-relaxed); margin-bottom: var(--pixel-space-2); color: var(--pixel-ink-soft);">
                    ${this.project.description}
                </p>

                <!-- XP Reward -->
                <div class="pixel-progress pixel-mb-2">
                    <div class="pixel-progress__bar" style="width: ${Math.floor(Math.random() * 40 + 60)}%;"></div>
                    <div class="pixel-progress__label" style="font-size: var(--pixel-font-xs);">+${Math.floor(Math.random() * 50 + 50)} XP</div>
                </div>

                <!-- Action Buttons -->
                <div class="pixel-flex pixel-gap-2">
                    <button class="pixel-btn pixel-btn--primary pixel-btn--sm read-review" style="flex: 1;">
                        🎮 Start Quest
                    </button>
                    <button class="pixel-btn pixel-btn--secondary pixel-btn--sm" style="min-width: auto;" title="Add to wishlist">
                        💾
                    </button>
                </div>
            </div>

            <!-- Resources -->
            <div class="pixel-flex pixel-flex-wrap pixel-gap-1" style="padding-top: var(--pixel-space-2); border-top: var(--pixel-border-thin);">
                ${resourcesHTML}
            </div>
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

        // Обработчик для клика по карточке (открытие по клику на карточку)
        card.addEventListener('click', (e) => {
            // Проверяем, что клик не по ссылке или кнопке - КАК В WEEKCARD
            if (!e.target.closest('a') && !e.target.closest('button')) {
                this._onReadReview();
            }
        });
        card.style.cursor = 'default';
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

        const elements = this.element.querySelectorAll('.project-card-title, .project-card-desc');
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
