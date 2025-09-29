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
        const jsonUrl = `${this.baseUrl}/web/infrastructure/data/index.json`;
        
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
        // Determine language mode based on current page path
        let isEnglish = false;
        try {
            const path = (typeof window !== 'undefined' && window.location && window.location.pathname) ? window.location.pathname : '';
            // If current page filename ends with _en.html, treat as English mode
            isEnglish = /_en\.html$/i.test(path);
        } catch (e) { /* no-op: default to RU */ }

        const fileName = isEnglish ? 'review_en.md' : 'review.md';
        const reviewUrl = `${this.baseUrl}/${yearNumber}/${weekId}/${fileName}`;

        // Helper to fetch with descriptive context
        const load = async (url, label) => {
            const response = await fetchWithRetry(url, {}, label);
            const text = await response.text();
            if (!text.trim()) {
                throw new Error('Статья пуста или не содержит контента');
            }
            return text;
        };

        // Try preferred language first; on failure, fallback to RU version
        try {
            return await load(reviewUrl, `статья "${yearNumber}/${weekId}"`);
        } catch (firstError) {
            if (isEnglish) {
                // Fallback to RU if EN not available
                const fallbackUrl = `${this.baseUrl}/${yearNumber}/${weekId}/review.md`;
                try {
                    return await load(fallbackUrl, `статья "${yearNumber}/${weekId}"`);
                } catch (fallbackError) {
                    throw fallbackError;
                }
            }
            throw firstError;
        }
        
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
