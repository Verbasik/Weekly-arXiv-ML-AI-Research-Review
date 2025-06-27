import { ErrorHandler, fetchWithRetry } from '../error/ErrorHandler.js';

/**
 * Agents Data Source - Infrastructure Layer
 * Реализация источника данных для Agents проектов
 */
export class AgentsDataSource {
    constructor(config) {
        this.githubRepo = config.githubRepo;
        this.githubBranch = config.githubBranch;
        this.baseUrl = `https://raw.githubusercontent.com/${this.githubRepo}/${this.githubBranch}`;
    }

    /**
     * Получает данные из agents-index.json
     */
    async fetchData() {
        const jsonUrl = `${this.baseUrl}/web/infrastructure/data/agents-index.json`;
        
        try {
            const response = await fetchWithRetry(jsonUrl, {}, 'данные agents проектов');
            const data = await response.json();
            
            if (!data.projects || !Array.isArray(data.projects)) {
                throw new Error('Invalid data format: projects array is missing');
            }
            
            console.log('✅ Agents data loaded successfully from GitHub');
            return data;
        } catch (error) {
            console.warn('⚠️ GitHub fetch failed, trying local fallback:', error.message);
            try {
                return await this._fetchLocalData();
            } catch (localError) {
                console.error('❌ Failed to load agents data from both GitHub and local:', localError);
                throw new Error(`Failed to fetch agents data: ${error.message}`);
            }
        }
    }

    /**
     * Загружает данные из локального файла (для разработки)
     */
    async _fetchLocalData() {
        try {
            const response = await fetch('/web/infrastructure/data/agents-index.json');
            
            if (!response.ok) {
                throw new Error(`Local file not found: HTTP ${response.status}`);
            }
            
            const data = await response.json();
            
            if (!data.projects || !Array.isArray(data.projects)) {
                throw new Error('Invalid local data format: projects array is missing');
            }
            
            console.log('✅ Agents data loaded successfully from local file');
            return data;
            
        } catch (error) {
            console.error('❌ Failed to load local agents data:', error);
            throw new Error(`Failed to fetch local agents data: ${error.message}`);
        }
    }

    /**
     * Получает markdown контент для проекта
     */
    async fetchMarkdown(projectId) {
        const reviewUrl = `${this.baseUrl}/agents-under-hood/${projectId}/README.md`;
        
        try {
            const response = await fetchWithRetry(reviewUrl, {}, `проект "${projectId}"`);
            let markdown = await response.text();
            
            if (!markdown.trim()) {
                throw new Error('Обзор проекта пуст или не содержит контента');
            }
            
            console.log(`✅ Markdown loaded successfully for ${projectId} from GitHub`);
            return markdown;
        } catch (error) {
            console.warn(`⚠️ GitHub markdown fetch failed for ${projectId}, trying local fallback:`, error.message);
            try {
                return await this._fetchLocalMarkdown(projectId);
            } catch (localError) {
                console.error(`❌ Failed to load markdown for ${projectId} from both GitHub and local:`, localError);
                throw new Error(`Failed to fetch markdown for ${projectId}: ${error.message}`);
            }
        }
    }

    /**
     * Загружает markdown из локального файла (для разработки)
     */
    async _fetchLocalMarkdown(projectId) {
        try {
            const response = await fetch(`/agents-under-hood/${projectId}/README.md`);
            
            if (!response.ok) {
                throw new Error(`Local markdown not found: HTTP ${response.status}`);
            }
            
            let markdown = await response.text();
            
            if (!markdown.trim()) {
                throw new Error('Local markdown is empty');
            }
            
            console.log(`✅ Markdown loaded successfully for ${projectId} from local file`);
            return markdown;
            
        } catch (error) {
            console.error(`❌ Failed to load local markdown for ${projectId}:`, error);
            throw new Error(`Failed to fetch local markdown for ${projectId}: ${error.message}`);
        }
    }

    /**
     * Проверяет доступность проекта
     */
    async checkProjectHealth(projectId) {
        try {
            const response = await fetch(`https://api.github.com/repos/${this.githubRepo}/contents/agents-under-hood/${projectId}?ref=${this.githubBranch}`);
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
            const response = await fetch(`https://api.github.com/repos/${this.githubRepo}/contents/agents-under-hood/${projectId}?ref=${this.githubBranch}`);
            if (!response.ok) {
                throw new Error(`GitHub API error: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            throw new Error(`Failed to fetch project info: ${error.message}`);
        }
    }

    /**
     * Получает список файлов в директории проекта
     */
    async getProjectContents(projectId) {
        try {
            const response = await fetch(`https://api.github.com/repos/${this.githubRepo}/contents/agents-under-hood/${projectId}?ref=${this.githubBranch}`);
            if (!response.ok) {
                throw new Error(`GitHub API error: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            throw new Error(`Failed to fetch project contents: ${error.message}`);
        }
    }
} 