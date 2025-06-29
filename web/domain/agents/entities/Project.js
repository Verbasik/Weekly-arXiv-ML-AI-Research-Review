/**
 * Project Entity - Domain Model for Agents
 * Представляет сущность "Проект" в домене Agents
 */
export class Project {
    constructor({
        id,
        title,
        date,
        tags = [],
        description = '',
        notebooks = 0,
        code = 0,
        notebook_path = null,
        code_path = null,
        github_stars = '',
        difficulty = 'Beginner',
        technologies = [],
        features = []
    }) {
        this.id = id;
        this.title = title;
        this.date = date;
        this.tags = tags;
        this.description = description;
        this.notebooks = notebooks;
        this.code = code;
        this.notebook_path = notebook_path;
        this.code_path = code_path;
        this.github_stars = github_stars;
        this.difficulty = difficulty;
        this.technologies = technologies;
        this.features = features;
    }

    /**
     * Проверяет, есть ли у проекта доступные ресурсы
     */
    hasResources() {
        return this.notebooks > 0 || this.code > 0;
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
     * Возвращает URL для обзора проекта
     */
    getReviewUrl(githubRepo, branch) {
        return `https://raw.githubusercontent.com/${githubRepo}/${branch}/${this.code_path}/README.md`;
    }

    /**
     * Проверяет, содержит ли проект указанный тег
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

        if (this.difficulty) {
            meta.push({
                icon: 'fas fa-chart-line',
                text: this.difficulty,
                type: 'difficulty'
            });
        }

        if (this.github_stars) {
            meta.push({
                icon: 'fas fa-star',
                text: this.github_stars,
                type: 'stars'
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
    getFormattedResources(githubRepo, branch, config = null) {
        const resources = [];

        // Paper
        if (this.code_path) {
            const paperText = `Paper`;
            const resource = {
                icon: 'far fa-file-alt',
                text: paperText,
                type: 'paper'
            };
            
            if (config && config.urls && config.urls.paper_base) {
                resource.url = config.urls.paper_base
                    .replace('{repo}', githubRepo)
                    .replace('{branch}', branch)
                    .replace('{code_path}', this.code_path);
            } else {
                resource.url = `https://raw.githubusercontent.com/${githubRepo}/${branch}/${this.code_path}/README.md`;
            }
            
            resources.push(resource);
        }

        // Notebooks - ссылка на GitHub tree как на главной странице
        if (this.notebooks !== undefined && this.notebooks > 0) {
            const notebooksText = `${this.notebooks} Notebook${this.notebooks !== 1 ? 's' : ''}`;
            const resource = {
                icon: 'far fa-file-code',
                text: notebooksText,
                type: 'notebooks'
            };
            
            if (this.notebook_path) {
                if (config && config.urls && config.urls.notebook_base) {
                    resource.url = config.urls.notebook_base
                        .replace('{repo}', githubRepo)
                        .replace('{branch}', branch)
                        .replace('{notebook_path}', this.notebook_path);
                } else {
                    resource.url = this.getNotebookUrl(githubRepo, branch);
                }
            }
            
            resources.push(resource);
        }

        // Code files - ссылка на GitHub tree как на главной странице
        if (this.code !== undefined && this.code > 0) {
            const codeText = `Code Files`;
            const resource = {
                icon: 'fas fa-code',
                text: codeText,
                type: 'code'
            };
            
            if (this.code_path) {
                if (config && config.urls && config.urls.code_base) {
                    resource.url = config.urls.code_base
                        .replace('{repo}', githubRepo)
                        .replace('{branch}', branch)  
                        .replace('{code_path}', this.code_path);
                } else {
                    resource.url = this.getCodeUrl(githubRepo, branch);
                }
            }
            
            resources.push(resource);
        }

        return resources;
    }

    /**
     * Проверяет валидность данных проекта
     */
    isValid() {
        return this.id && this.title && this.description;
    }

    /**
     * Возвращает уникальный идентификатор проекта
     */
    getId() {
        return this.id;
    }

    /**
     * Возвращает краткое описание проекта
     */
    getSummary() {
        const maxLength = 150;
        if (this.description.length <= maxLength) {
            return this.description;
        }
        return this.description.substring(0, maxLength) + '...';
    }

    /**
     * Возвращает цвет индикатора сложности
     */
    getDifficultyColor() {
        const colors = {
            'Beginner': '#4CAF50',
            'Intermediate': '#FF9800', 
            'Advanced': '#F44336'
        };
        return colors[this.difficulty] || '#9E9E9E';
    }
} 