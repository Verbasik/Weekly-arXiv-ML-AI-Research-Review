import { ErrorHandler, fetchWithRetry } from '../error/ErrorHandler.js';

/**
 * GitHub Data Source - Infrastructure Layer
 * Реализация источника данных для GitHub
 */
export class GitHubDataSource {
    constructor(config) {
        this.githubRepo = config.githubRepo;
        this.githubBranch = config.githubBranch;
        // Default base URL uses provided branch; per-request methods may override branch by language
        this.baseUrl = `https://raw.githubusercontent.com/${this.githubRepo}/${this.githubBranch}`;
    }

    /**
     * Получает данные из index.json
     */
    async fetchData() {
        // Determine language mode based on current page path
        let isEnglish = false;
        try {
            const path = (typeof window !== 'undefined' && window.location && window.location.pathname) ? window.location.pathname : '';
            // Treat pages ending with -en.html or _en.html as English mode
            isEnglish = /(?:-|_)en\.html$/i.test(path);
        } catch (e) { /* no-op: default to RU */ }

        // Select branch by language: EN → main-en; otherwise use configured branch
        const branch = isEnglish ? 'main-en' : this.githubBranch;
        const langBase = `https://raw.githubusercontent.com/${this.githubRepo}/${branch}`;

        // Prefer EN-named index in EN branch, then its alternatives, then plain index.json in EN branch;
        // Finally, as a last resort, fallback to RU branch main/index.json
        const primaryUrl = `${langBase}/web/infrastructure/data/${isEnglish ? 'index-en.json' : 'index.json'}`;
        const altEnUrl = `${langBase}/web/infrastructure/data/index_en.json`;
        const enPlainUrl = `${langBase}/web/infrastructure/data/index.json`;
        const ruFallbackUrl = `https://raw.githubusercontent.com/${this.githubRepo}/main/web/infrastructure/data/index.json`;
        
        try {
            let response = await fetchWithRetry(primaryUrl, {}, 'данные статей');
            if (!response.ok && isEnglish) {
                response = await fetchWithRetry(altEnUrl, {}, 'данные статей');
                if (!response.ok) {
                    response = await fetchWithRetry(enPlainUrl, {}, 'данные статей');
                    if (!response.ok) {
                        response = await fetchWithRetry(ruFallbackUrl, {}, 'данные статей');
                    }
                }
            }

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
            // Treat pages ending with -en.html or _en.html as English mode
            isEnglish = /(?:-|_)en\.html$/i.test(path);
        } catch (e) { /* no-op: default to RU */ }

        // Select branch by language: EN → main-en; otherwise use configured branch
        const branch = isEnglish ? 'main-en' : this.githubBranch;
        const langBase = `https://raw.githubusercontent.com/${this.githubRepo}/${branch}`;

        // In EN branch we store unified filenames: review.md
        const preferred = 'review.md';
        const altEn = 'review-en.md';
        const altEn2 = 'review_en.md';
        const preferredUrl = `${langBase}/${yearNumber}/${weekId}/${preferred}`;

        // Helper to fetch with descriptive context
        const load = async (url, label) => {
            const response = await fetchWithRetry(url, {}, label);
            const text = await response.text();
            if (!text.trim()) {
                throw new Error('Статья пуста или не содержит контента');
            }
            return text;
        };

        // Try preferred name first; on EN pages also try legacy EN names within EN branch;
        // finally fallback to RU main branch review.md
        try {
            return await load(preferredUrl, `статья "${yearNumber}/${weekId}"`);
        } catch (firstError) {
            if (isEnglish) {
                // Try legacy EN filenames inside EN branch
                try {
                    return await load(`${langBase}/${yearNumber}/${weekId}/${altEn}`, `статья "${yearNumber}/${weekId}"`);
                } catch (_) {
                    try {
                        return await load(`${langBase}/${yearNumber}/${weekId}/${altEn2}`, `статья "${yearNumber}/${weekId}"`);
                    } catch (_) { /* continue to RU fallback */ }
                }
                // Fallback to RU branch main
                const ruUrl = `https://raw.githubusercontent.com/${this.githubRepo}/main/${yearNumber}/${weekId}/review.md`;
                return await load(ruUrl, `статья "${yearNumber}/${weekId}"`);
            }
            // Non-EN: rethrow original error
            throw firstError;
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
