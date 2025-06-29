/**
 * Research Service - Domain Service
 * Содержит бизнес-логику для работы с исследованиями
 */
export class ResearchService {
    constructor(repository) {
        this.repository = repository;
    }

    /**
     * Получает все годы с неделями
     */
    async getAllResearchData() {
        return await this.repository.getAllYears();
    }

    /**
     * Получает данные для конкретного года
     */
    async getYearData(yearNumber) {
        const year = await this.repository.getYear(yearNumber);
        if (!year) {
            throw new Error(`Year ${yearNumber} not found`);
        }
        return year;
    }

    /**
     * Получает данные недели с проверкой существования
     */
    async getWeekData(yearNumber, weekId) {
        const week = await this.repository.getWeek(yearNumber, weekId);
        if (!week) {
            throw new Error(`Week ${weekId} in year ${yearNumber} not found`);
        }
        return { year: yearNumber, week };
    }

    /**
     * Получает markdown контент для недели
     */
    async getWeekMarkdown(yearNumber, weekId) {
        // Сначала проверяем, что неделя существует
        await this.getWeekData(yearNumber, weekId);
        
        // Затем получаем markdown
        return await this.repository.getWeekMarkdown(yearNumber, weekId);
    }

    /**
     * Получает популярные теги
     */
    async getPopularTags(limit = 10) {
        const years = await this.repository.getAllYears();
        const tagCounts = new Map();
        
        years.forEach(year => {
            year.getWeeks().forEach(week => {
                week.tags.forEach(tag => {
                    tagCounts.set(tag, (tagCounts.get(tag) || 0) + 1);
                });
            });
        });
        
        // Сортируем по популярности и возвращаем топ N
        return Array.from(tagCounts.entries())
            .sort((a, b) => b[1] - a[1])
            .slice(0, limit)
            .map(([tag, count]) => ({ tag, count }));
    }

    /**
     * Получает trending статьи (с наибольшим количеством ресурсов)
     */
    async getTrendingPapers(limit = 5) {
        const years = await this.repository.getAllYears();
        const allWeeks = [];
        
        years.forEach(year => {
            year.getWeeks().forEach(week => {
                allWeeks.push({
                    year: year.year,
                    week: week,
                    score: this._calculateTrendingScore(week)
                });
            });
        });
        
        return allWeeks
            .sort((a, b) => b.score - a.score)
            .slice(0, limit);
    }

    /**
     * Поиск по исследованиям
     */
    async searchResearch(query) {
        if (!query || query.trim().length < 2) {
            throw new Error('Search query must be at least 2 characters long');
        }
        
        return await this.repository.searchWeeks(query.trim());
    }

    /**
     * Фильтрация по тегу
     */
    async filterByTag(tag) {
        if (!tag) {
            throw new Error('Tag is required for filtering');
        }
        
        return await this.repository.getWeeksByTag(tag);
    }

    /**
     * Получает статистику исследований
     */
    async getResearchStatistics() {
        return await this.repository.getStatistics();
    }

    /**
     * Получает последние добавленные недели
     */
    async getLatestWeeks(limit = 5) {
        const years = await this.repository.getAllYears();
        const allWeeks = [];
        
        years.forEach(year => {
            year.getWeeks().forEach(week => {
                allWeeks.push({
                    year: year.year,
                    week: week
                });
            });
        });
        
        // Сортируем по году и номеру недели (последние сначала)
        return allWeeks
            .sort((a, b) => {
                if (a.year !== b.year) {
                    return b.year - a.year;
                }
                const weekNumA = parseInt(a.week.week.match(/\d+/)?.[0] || '0');
                const weekNumB = parseInt(b.week.week.match(/\d+/)?.[0] || '0');
                return weekNumB - weekNumA;
            })
            .slice(0, limit);
    }

    /**
     * Валидирует данные недели
     */
    validateWeekData(weekData) {
        const errors = [];
        
        if (!weekData.week) {
            errors.push('Week identifier is required');
        }
        
        if (!weekData.title) {
            errors.push('Week title is required');
        }
        
        if (!weekData.description) {
            errors.push('Week description is required');
        }
        
        if (weekData.tags && !Array.isArray(weekData.tags)) {
            errors.push('Tags must be an array');
        }
        
        return {
            isValid: errors.length === 0,
            errors
        };
    }

    /**
     * Получает рекомендации на основе тегов недели
     */
    async getRecommendations(yearNumber, weekId, limit = 3) {
        const { week } = await this.getWeekData(yearNumber, weekId);
        const recommendations = [];
        
        // Находим недели с похожими тегами
        for (const tag of week.tags) {
            const similarWeeks = await this.repository.getWeeksByTag(tag);
            
            similarWeeks.forEach(({ year, week: similarWeek }) => {
                // Исключаем текущую неделю
                if (year === yearNumber && similarWeek.getId() === weekId) {
                    return;
                }
                
                const similarity = this._calculateSimilarity(week, similarWeek);
                recommendations.push({
                    year,
                    week: similarWeek,
                    similarity
                });
            });
        }
        
        // Убираем дубликаты и сортируем по схожести
        const uniqueRecommendations = recommendations.reduce((acc, current) => {
            const existing = acc.find(item => 
                item.year === current.year && 
                item.week.getId() === current.week.getId()
            );
            
            if (!existing) {
                acc.push(current);
            } else if (current.similarity > existing.similarity) {
                existing.similarity = current.similarity;
            }
            
            return acc;
        }, []);
        
        return uniqueRecommendations
            .sort((a, b) => b.similarity - a.similarity)
            .slice(0, limit);
    }

    /**
     * Вычисляет trending score для недели
     */
    _calculateTrendingScore(week) {
        let score = 0;
        
        // Баллы за ресурсы
        score += (week.papers || 0) * 3;
        score += (week.notebooks || 0) * 2;
        score += (week.code || 0) * 1;
        
        // Баллы за популярные теги
        const popularTags = ['LLMs', 'Deep Learning', 'Transformers', 'AI'];
        week.tags.forEach(tag => {
            if (popularTags.includes(tag)) {
                score += 2;
            }
        });
        
        return score;
    }

    /**
     * Вычисляет схожесть между неделями
     */
    _calculateSimilarity(week1, week2) {
        const tags1 = new Set(week1.tags);
        const tags2 = new Set(week2.tags);
        
        // Пересечение тегов
        const intersection = new Set([...tags1].filter(tag => tags2.has(tag)));
        const union = new Set([...tags1, ...tags2]);
        
        // Коэффициент Жаккара
        return intersection.size / union.size;
    }
} 