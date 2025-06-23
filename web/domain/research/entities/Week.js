/**
 * Week Entity - Domain Model
 * Представляет сущность "Неделя исследований" в домене Research
 */
export class Week {
    constructor({
        week,
        title,
        date,
        tags = [],
        description = '',
        papers = 0,
        notebooks = 0,
        code = 0,
        notebook_path = null,
        code_path = null
    }) {
        this.week = week;
        this.title = title;
        this.date = date;
        this.tags = tags;
        this.description = description;
        this.papers = papers;
        this.notebooks = notebooks;
        this.code = code;
        this.notebook_path = notebook_path;
        this.code_path = code_path;
    }

    /**
     * Проверяет, есть ли у недели доступные ресурсы
     */
    hasResources() {
        return this.papers > 0 || this.notebooks > 0 || this.code > 0;
    }

    /**
     * Возвращает URL для ноутбуков
     */
    getNotebookUrl(githubRepo, branch) {
        if (!this.notebook_path) return null;
        return `https://github.com/${githubRepo}/tree/${branch}/${this.notebook_path}`;
    }

    /**
     * Возвращает URL для кода
     */
    getCodeUrl(githubRepo, branch) {
        if (!this.code_path) return null;
        return `https://github.com/${githubRepo}/tree/${branch}/${this.code_path}`;
    }

    /**
     * Возвращает URL для обзора статьи
     */
    getReviewUrl(githubRepo, branch, year) {
        return `https://raw.githubusercontent.com/${githubRepo}/${branch}/${year}/${this.week}/review.md`;
    }

    /**
     * Проверяет, содержит ли неделя указанный тег
     */
    hasTag(tag) {
        return this.tags.includes(tag);
    }

    /**
     * Форматирует метаданные для отображения
     */
    getFormattedMeta() {
        const meta = [];
        
        if (this.date) {
            meta.push({
                icon: 'far fa-calendar',
                text: this.date
            });
        }
        
        this.tags.forEach(tag => {
            meta.push({
                icon: 'fas fa-tag',
                text: tag,
                type: 'tag'
            });
        });
        
        return meta;
    }

    /**
     * Форматирует ресурсы для отображения в футере карточки
     */
    getFormattedResources(githubRepo, branch) {
        const resources = [];

        // Papers
        if (this.papers !== undefined) {
            const paperText = `${this.papers} Paper${this.papers !== 1 ? 's' : ''}`;
            resources.push({
                icon: 'far fa-file-alt',
                text: paperText,
                type: 'papers'
            });
        }

        // Notebooks
        if (this.notebooks !== undefined) {
            const notebooksText = `${this.notebooks} Notebook${this.notebooks !== 1 ? 's' : ''}`;
            const resource = {
                icon: 'far fa-file-code',
                text: notebooksText,
                type: 'notebooks'
            };
            
            if (this.notebook_path) {
                resource.url = this.getNotebookUrl(githubRepo, branch);
            }
            
            resources.push(resource);
        }

        // Code files
        if (this.code !== undefined) {
            const codeText = `${this.code} Code${this.code !== 1 ? ' files' : ''}`;
            const resource = {
                icon: 'fas fa-code',
                text: codeText,
                type: 'code'
            };
            
            if (this.code_path) {
                resource.url = this.getCodeUrl(githubRepo, branch);
            }
            
            resources.push(resource);
        }

        return resources;
    }

    /**
     * Проверяет валидность данных недели
     */
    isValid() {
        return this.week && this.title && this.description;
    }

    /**
     * Возвращает уникальный идентификатор недели
     */
    getId() {
        return this.week;
    }

    /**
     * Возвращает краткое описание недели
     */
    getSummary() {
        const maxLength = 150;
        if (this.description.length <= maxLength) {
            return this.description;
        }
        return this.description.substring(0, maxLength) + '...';
    }
} 