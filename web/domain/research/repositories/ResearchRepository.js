import { Week } from '../entities/Week.js';
import { Year } from '../entities/Year.js';

/**
 * Research Repository - Data Access Layer
 * Абстракция для работы с данными исследований
 */
export class ResearchRepository {
    constructor(dataSource) {
        this.dataSource = dataSource;
        this._yearsCache = null;
    }

    /**
     * Получает все годы с неделями
     */
    async getAllYears() {
        try {
            if (this._yearsCache) {
                return this._yearsCache;
            }
            const data = await this.dataSource.fetchData();
            this._yearsCache = this._mapToYearEntities(data.years);
            return this._yearsCache;
        } catch (error) {
            throw new Error(`Failed to fetch research data: ${error.message}`);
        }
    }

    /**
     * Получает конкретный год
     */
    async getYear(yearNumber) {
        const years = await this.getAllYears();
        return years.find(year => year.year === yearNumber);
    }

    /**
     * Получает конкретную неделю
     */
    async getWeek(yearNumber, weekId) {
        const year = await this.getYear(yearNumber);
        return year?.getWeek(weekId);
    }

    /**
     * Получает все доступные теги
     */
    async getAllTags() {
        const years = await this.getAllYears();
        const allTags = years.flatMap(year => year.getAllTags());
        return [...new Set(allTags)];
    }

    /**
     * Фильтрует недели по тегу
     */
    async getWeeksByTag(tag) {
        const years = await this.getAllYears();
        const weeksByTag = [];
        
        years.forEach(year => {
            const weeks = year.getWeeksByTag(tag);
            weeks.forEach(week => {
                weeksByTag.push({
                    year: year.year,
                    week: week
                });
            });
        });
        
        return weeksByTag;
    }

    /**
     * Получает статистику по всем годам
     */
    async getStatistics() {
        const years = await this.getAllYears();
        return {
            totalYears: years.length,
            totalWeeks: years.reduce((total, year) => total + year.getWeeksCount(), 0),
            totalPapers: years.reduce((total, year) => total + year.getTotalPapers(), 0),
            totalNotebooks: years.reduce((total, year) => total + year.getTotalNotebooks(), 0),
            yearStatistics: years.map(year => year.getStatistics())
        };
    }

    /**
     * Получает контент markdown для недели
     */
    async getWeekMarkdown(yearNumber, weekId) {
        try {
            return await this.dataSource.fetchMarkdown(yearNumber, weekId);
        } catch (error) {
            throw new Error(`Failed to fetch markdown for ${yearNumber}/${weekId}: ${error.message}`);
        }
    }

    /**
     * Поиск недель по запросу
     */
    async searchWeeks(query) {
        const years = await this.getAllYears();
        const results = [];
        
        const searchTerm = query.toLowerCase();
        
        years.forEach(year => {
            year.getWeeks().forEach(week => {
                const matchesTitle = week.title.toLowerCase().includes(searchTerm);
                const matchesDescription = week.description.toLowerCase().includes(searchTerm);
                const matchesTags = week.tags.some(tag => tag.toLowerCase().includes(searchTerm));
                
                if (matchesTitle || matchesDescription || matchesTags) {
                    results.push({
                        year: year.year,
                        week: week,
                        relevance: this._calculateRelevance(week, searchTerm)
                    });
                }
            });
        });
        
        // Сортируем по релевантности
        return results.sort((a, b) => b.relevance - a.relevance);
    }

    /**
     * Преобразует сырые данные в сущности Year
     */
    _mapToYearEntities(yearsData) {
        return yearsData.map(yearData => {
            const weeks = yearData.weeks.map(weekData => new Week(weekData));
            return new Year(yearData.year, weeks);
        });
    }

    /**
     * Вычисляет релевантность поиска
     */
    _calculateRelevance(week, searchTerm) {
        let relevance = 0;
        
        // Вес для совпадения в заголовке
        if (week.title.toLowerCase().includes(searchTerm)) {
            relevance += 10;
        }
        
        // Вес для совпадения в описании
        if (week.description.toLowerCase().includes(searchTerm)) {
            relevance += 5;
        }
        
        // Вес для совпадения в тегах
        week.tags.forEach(tag => {
            if (tag.toLowerCase().includes(searchTerm)) {
                relevance += 7;
            }
        });
        
        return relevance;
    }
} 
