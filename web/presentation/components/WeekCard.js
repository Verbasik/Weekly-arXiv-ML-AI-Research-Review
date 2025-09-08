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
        card.className = 'pixel-card week-card';
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

        // Convert FontAwesome icons to emojis for pixel theme
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
                return `<a href="${resource.url}" target="_blank" class="pixel-btn pixel-btn--sm" style="font-size: var(--pixel-font-xs);">${emoji} ${resource.text}</a>`;
            }
            return `<span class="pixel-badge" data-icon="${emoji}">${resource.text}</span>`;
        }).join('');

        // Generate random quest difficulty
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
                    <div style="font-size: 2rem;">📜</div>
                    <div>
                        <h3 style="font-family: var(--pixel-font-display); font-size: var(--pixel-font-base); margin-bottom: var(--px-unit-half); color: var(--pixel-ink);">
                            ${this.week.title}
                        </h3>
                        <div class="pixel-badge pixel-badge--${difficulty.color}" data-icon="${difficulty.emoji}">
                            ${difficulty.level} Quest
                        </div>
                    </div>
                </div>
                <div style="font-size: 1.5rem;">⚔️</div>
            </div>

            <!-- Quest Details -->
            <div class="pixel-mb-3">
                <div class="pixel-flex pixel-flex-wrap pixel-gap-1 pixel-mb-2">
                    ${metaHTML}
                </div>
                <p style="font-size: var(--pixel-font-sm); line-height: var(--pixel-line-relaxed); margin-bottom: var(--pixel-space-2); color: var(--pixel-ink-soft);">
                    ${this.week.getSummary()}
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

            <!-- Resources (Inventory Items) -->
            <div class="pixel-flex pixel-flex-wrap pixel-gap-1" style="padding-top: var(--pixel-space-2); border-top: var(--pixel-border-thin);">
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
                e.stopPropagation(); // Предотвращаем всплытие
                this._onReadReview();
            });
        }

        // Убираем обработчик клика по карточке - только кнопка должна открывать обзор
        // card.addEventListener('click', ...) - удалено

        // Убираем курсор pointer с карточки, оставляем только на кнопке
        card.style.cursor = 'default';
    }

    /**
     * Обработчик открытия обзора
     */
    _onReadReview() {
        // Проверяем, есть ли глобальный readingModal
        if (window.readingModal) {
            const title = `${this.year} Week ${this.week.getId()}: ${this.week.title}`;
            
            // Открываем полноэкранное модальное окно
            window.readingModal.open(title);
            
            // Симулируем загрузку контента (здесь будет реальная загрузка)
            setTimeout(() => {
                const content = `
                    <h1>${this.week.title}</h1>
                    <h2>Техническое описание</h2>
                    <p>Это детальный технический обзор статьи "${this.week.title}" за ${this.year} год, неделя ${this.week.getId()}.</p>
                    
                    <h3>Основные концепции</h3>
                    <ul>
                        <li>Инновационный подход к решению задач</li>
                        <li>Применение современных алгоритмов</li>
                        <li>Практические результаты исследования</li>
                    </ul>
                    
                    <h3>Методология</h3>
                    <p>В данной работе используются передовые методы анализа данных и машинного обучения.</p>
                    
                    <blockquote>
                        <p>Этот обзор демонстрирует возможности полноэкранного режима чтения с оптимальной типографикой и удобным управлением.</p>
                    </blockquote>
                    
                    <h3>Заключение</h3>
                    <p>Представленное исследование открывает новые горизонты в области ИИ и машинного обучения.</p>
                `;
                window.readingModal.setContent(content);
            }, 1500);
        } else {
            // Fallback - создаем кастомное событие для старого модального окна
            const event = new CustomEvent('openReview', {
                detail: {
                    year: this.year,
                    weekId: this.week.getId(),
                    title: this.week.title
                }
            });
            
            document.dispatchEvent(event);
        }
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