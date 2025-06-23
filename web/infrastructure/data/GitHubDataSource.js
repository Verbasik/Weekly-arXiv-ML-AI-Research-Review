import { ErrorHandler, fetchWithRetry } from '../error/ErrorHandler.js';

/**
 * GitHub Data Source - Infrastructure Layer
 * Реализация источника данных для GitHub
 */
export class GitHubDataSource {
    constructor(config) {
        this.githubRepo = config.githubRepo;
        this.githubBranch = config.githubBranch;
        this.baseUrl = `https://raw.githubusercontent.com/${this.githubRepo}/${this.githubBranch}`;
    }

    /**
     * Получает данные из index.json
     */
    async fetchData() {
        const jsonUrl = `${this.baseUrl}/web/index.json`;
        
        try {
            const response = await fetchWithRetry(jsonUrl, {}, 'данные статей');
            const data = await response.json();
            
            if (!data.years || !Array.isArray(data.years)) {
                throw new Error('Invalid data format: years array is missing');
            }
            
            return data;
        } catch (error) {
            throw new Error(`Failed to fetch research data: ${error.message}`);
        }
    }

    /**
     * Получает markdown контент для недели
     */
    async fetchMarkdown(yearNumber, weekId) {
        const reviewUrl = `${this.baseUrl}/${yearNumber}/${weekId}/review.md`;
        
        try {
            const response = await fetchWithRetry(reviewUrl, {}, `статья "${yearNumber}/${weekId}"`);
            let markdown = await response.text();
            
            if (!markdown.trim()) {
                throw new Error('Статья пуста или не содержит контента');
            }
            
            return markdown;
        } catch (error) {
            throw new Error(`Failed to fetch markdown for ${yearNumber}/${weekId}: ${error.message}`);
        }
    }

    /**
     * Проверяет доступность GitHub API
     */
    async checkHealth() {
        try {
            const response = await fetch(`https://api.github.com/repos/${this.githubRepo}`);
            return response.ok;
        } catch (error) {
            return false;
        }
    }

    /**
     * Получает информацию о репозитории
     */
    async getRepositoryInfo() {
        try {
            const response = await fetch(`https://api.github.com/repos/${this.githubRepo}`);
            if (!response.ok) {
                throw new Error(`GitHub API error: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            throw new Error(`Failed to fetch repository info: ${error.message}`);
        }
    }

    /**
     * Получает список файлов в директории
     */
    async getDirectoryContents(path) {
        try {
            const response = await fetch(`https://api.github.com/repos/${this.githubRepo}/contents/${path}?ref=${this.githubBranch}`);
            if (!response.ok) {
                throw new Error(`GitHub API error: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            throw new Error(`Failed to fetch directory contents: ${error.message}`);
        }
    }
} 