import { ErrorHandler, fetchWithRetry } from '../error/ErrorHandler.js';

/**
 * Agents Data Source - Infrastructure Layer
 * Источник данных для получения информации об agents проектах
 */
export class AgentsDataSource {
    constructor(config) {
        this.githubRepo = config.githubRepo;
        this.githubBranch = config.githubBranch;
        this.config = null;
        this.configLoaded = false;
        // Запускаем загрузку конфигурации, но не блокируем конструктор
        this._initializeConfig();
    }

    /**
     * Инициализирует загрузку конфигурации
     */
    async _initializeConfig() {
        try {
            await this._loadConfig();
            this.configLoaded = true;
        } catch (error) {
            console.warn('Failed to initialize config:', error);
            this.configLoaded = true; // Отмечаем как загруженную с fallback
        }
    }

    /**
     * Загружает конфигурацию
     */
    async _loadConfig() {
        try {
            const response = await fetch('web/infrastructure/data/agents-config.json');
            const configData = await response.json();
            this.config = configData.config;
        } catch (error) {
            console.warn('Failed to load agents config, using defaults:', error);
            // Fallback конфигурация
            this.config = {
                github: {
                    repo: this.githubRepo,
                    branch: this.githubBranch
                },
                urls: {
                    paper_base: `https://raw.githubusercontent.com/${this.githubRepo}/${this.githubBranch}/{code_path}/README.md`,
                    notebook_base: `https://github.com/${this.githubRepo}/tree/${this.githubBranch}/{notebook_path}`,
                    code_base: `https://github.com/${this.githubRepo}/tree/${this.githubBranch}/{code_path}`
                },
                buttons: {
                    paper: {
                        text: "Paper",
                        icon: "far fa-file-alt",
                        behavior: "modal",
                        url_template: "paper_base"
                    },
                    notebooks: {
                        text: "{count} Notebook{plural}",
                        icon: "far fa-file-code", 
                        behavior: "redirect",
                        url_template: "notebook_base"
                    },
                    code: {
                        text: "Code Files",
                        icon: "fas fa-code",
                        behavior: "redirect", 
                        url_template: "code_base"
                    }
                }
            };
        }
    }

    /**
     * Получает конфигурацию (с ожиданием загрузки если нужно)
     */
    async getConfig() {
        // Ждем завершения загрузки конфигурации
        if (!this.configLoaded) {
            await this._initializeConfig();
        }
        return this.config;
    }

    /**
     * Получает конфигурацию синхронно (может вернуть null если не загружена)
     */
    getConfigSync() {
        return this.config;
    }

    /**
     * Получает данные agents проектов
     */
    async fetchData() {
        try {
            // Сначала пробуем загрузить удаленно
            const response = await fetch(`https://raw.githubusercontent.com/${this.githubRepo}/${this.githubBranch}/web/infrastructure/data/agents-index.json`);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            return await response.json();
        } catch (error) {
            console.warn('Failed to fetch remote agents data, trying local:', error.message);
            return await this._fetchLocalData();
        }
    }

    /**
     * Резервная загрузка локальных данных
     */
    async _fetchLocalData() {
        try {
            const response = await fetch('web/infrastructure/data/agents-index.json');
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: Failed to load local agents data`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('Failed to load local agents data:', error);
            // Возвращаем пустую структуру как fallback
            return {
                projects: [],
                metadata: {
                    total_projects: 0,
                    last_updated: new Date().toISOString(),
                    categories: []
                }
            };
        }
    }

    /**
     * Получает markdown контент для проекта
     */
    async fetchMarkdown(projectId) {
        try {
            // Сначала пробуем удаленно
            const response = await fetch(`https://raw.githubusercontent.com/${this.githubRepo}/${this.githubBranch}/agents-under-hood/${projectId}/README.md`);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            return await response.text();
        } catch (error) {
            console.warn('Failed to fetch remote markdown, trying local:', error.message);
            return await this._fetchLocalMarkdown(projectId);
        }
    }

    /**
     * Резервная загрузка локального markdown
     */
    async _fetchLocalMarkdown(projectId) {
        try {
            // Пробуем найти локальный файл
            const response = await fetch(`../agents-under-hood/${projectId}/README.md`);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: Local markdown not found`);
            }
            
            return await response.text();
        } catch (error) {
            console.error(`Failed to load markdown for ${projectId}:`, error);
            
            // Fallback контент
            return `# ${projectId}
            
Контент проекта временно недоступен.

Попробуйте:
1. Обновить страницу
2. Проверить подключение к интернету
3. Посетить [репозиторий проекта](https://github.com/${this.githubRepo})

Ошибка: ${error.message}`;
        }
    }

    /**
     * Проверяет доступность проекта
     */
    async checkProjectHealth(projectId) {
        try {
            const response = await fetch(`https://api.github.com/repos/${this.githubRepo}/contents/agents-under-hood/${projectId}`, {
                method: 'HEAD'
            });
            return response.ok;
        } catch (error) {
            return false;
        }
    }

    /**
     * Получает информацию о проекте из GitHub API
     */
    async getProjectInfo(projectId) {
        try {
            const response = await fetch(`https://api.github.com/repos/${this.githubRepo}/contents/agents-under-hood/${projectId}`);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            return await response.json();
        } catch (error) {
            console.warn(`Failed to fetch project info for ${projectId}:`, error.message);
            return null;
        }
    }

    /**
     * Получает список файлов проекта
     */
    async getProjectContents(projectId) {
        try {
            const info = await this.getProjectInfo(projectId);
            if (info && Array.isArray(info)) {
                return info.filter(item => item.type === 'file').map(file => ({
                    name: file.name,
                    path: file.path,
                    download_url: file.download_url
                }));
            }
            return [];
        } catch (error) {
            console.warn(`Failed to get project contents for ${projectId}:`, error.message);
            return [];
        }
    }
} 