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
        const isEn = (typeof document !== 'undefined') && document.documentElement?.getAttribute('lang') === 'en';
        const file = isEn ? 'index.en.json' : 'index.json';
        const jsonUrl = `${this.baseUrl}/web/infrastructure/data/${file}`;
        
        try {
            const response = await fetchWithRetry(jsonUrl, {}, 'данные статей');
            const data = await response.json();
            
            if (!data.years || !Array.isArray(data.years)) {
                throw new Error('Invalid data format: years array is missing');
            }
            
            return data;
        } catch (error) {
            // Fallback to RU index if EN missing/unavailable
            if (isEn) {
                try {
                    const fallbackUrl = `${this.baseUrl}/web/infrastructure/data/index.json`;
                    const resp = await fetchWithRetry(fallbackUrl, {}, 'данные статей (fallback)');
                    const data = await resp.json();
                    if (!data.years || !Array.isArray(data.years)) {
                        throw new Error('Invalid data format (fallback): years array is missing');
                    }
                    return data;
                } catch (err) {
                    throw new Error(`Failed to fetch research data (en and fallback): ${err.message}`);
                }
            }
            throw new Error(`Failed to fetch research data: ${error.message}`);
        }
    }

    /**
     * Получает markdown контент для недели
     */
    async fetchMarkdown(yearNumber, weekId) {
        const basePath = `${this.baseUrl}/${yearNumber}/${weekId}`;
        const isEn = (typeof document !== 'undefined') && document.documentElement?.getAttribute('lang') === 'en';
        const tryOrder = isEn ? ['review.en.md', 'review.md'] : ['review.md'];
        let lastError;
        for (const file of tryOrder) {
            const url = `${basePath}/${file}`;
            try {
                const response = await fetchWithRetry(url, {}, `статья "${yearNumber}/${weekId}"`);
                const markdown = await response.text();
                if (!markdown.trim()) throw new Error('Статья пуста или не содержит контента');
                // If EN requested but RU used, prepend fallback notice block
                if (isEn && file === 'review.md') {
                    const notice = `\n<div class=\"pixel-card pixel-card--warning\" style=\"margin-bottom: var(--pixel-space-2);\"><strong>EN unavailable:</strong> showing Russian version. Want to help translate? <a href=\"https://github.com/${this.githubRepo}\" target=\"_blank\">Contribute</a>.</div>\n`;
                    return notice + markdown;
                }
                return markdown;
            } catch (err) {
                lastError = err;
                // try next
            }
        }
        throw new Error(`Failed to fetch markdown for ${yearNumber}/${weekId}: ${lastError?.message || 'unknown error'}`);
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
