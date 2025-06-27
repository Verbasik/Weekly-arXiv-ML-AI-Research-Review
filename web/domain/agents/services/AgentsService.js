/**
 * Agents Service - Domain Service
 * Содержит бизнес-логику для работы с agents проектами
 */
export class AgentsService {
    constructor(repository) {
        this.repository = repository;
    }

    /**
     * Получает все проекты
     */
    async getAllProjects() {
        return await this.repository.getAllProjects();
    }

    /**
     * Получает данные для конкретного проекта
     */
    async getProjectData(projectId) {
        const project = await this.repository.getProject(projectId);
        if (!project) {
            throw new Error(`Project ${projectId} not found`);
        }
        return { project };
    }

    /**
     * Получает markdown контент для проекта
     */
    async getProjectMarkdown(projectId) {
        // Сначала проверяем, что проект существует
        await this.getProjectData(projectId);
        
        // Затем получаем markdown
        return await this.repository.getProjectMarkdown(projectId);
    }

    /**
     * Получает популярные теги
     */
    async getPopularTags(limit = 10) {
        const projects = await this.repository.getAllProjects();
        const tagCounts = new Map();
        
        projects.forEach(project => {
            project.tags.forEach(tag => {
                tagCounts.set(tag, (tagCounts.get(tag) || 0) + 1);
            });
        });
        
        // Сортируем по популярности и возвращаем топ N
        return Array.from(tagCounts.entries())
            .sort((a, b) => b[1] - a[1])
            .slice(0, limit)
            .map(([tag, count]) => ({ tag, count }));
    }

    /**
     * Получает featured проекты
     */
    async getFeaturedProjects(limit = 5) {
        const projects = await this.repository.getAllProjects();
        
        return projects
            .sort((a, b) => this._calculateProjectScore(b) - this._calculateProjectScore(a))
            .slice(0, limit);
    }

    /**
     * Поиск по проектам
     */
    async searchProjects(query) {
        if (!query || query.trim().length < 2) {
            throw new Error('Search query must be at least 2 characters long');
        }
        
        return await this.repository.searchProjects(query.trim());
    }

    /**
     * Фильтрация по тегу
     */
    async filterByTag(tag) {
        if (!tag) {
            throw new Error('Tag is required for filtering');
        }
        
        return await this.repository.getProjectsByTag(tag);
    }

    /**
     * Фильтрация по сложности
     */
    async filterByDifficulty(difficulty) {
        if (!difficulty) {
            throw new Error('Difficulty is required for filtering');
        }
        
        return await this.repository.getProjectsByDifficulty(difficulty);
    }

    /**
     * Фильтрация по технологии
     */
    async filterByTechnology(technology) {
        if (!technology) {
            throw new Error('Technology is required for filtering');
        }
        
        return await this.repository.getProjectsByTechnology(technology);
    }

    /**
     * Получает статистику проектов
     */
    async getProjectsStatistics() {
        return await this.repository.getStatistics();
    }

    /**
     * Получает последние добавленные проекты
     */
    async getLatestProjects(limit = 5) {
        const projects = await this.repository.getAllProjects();
        
        // Сортируем по дате (если есть) или по порядку добавления
        return projects
            .sort((a, b) => {
                // Простая сортировка по году в date поле
                const yearA = parseInt(a.date) || 0;
                const yearB = parseInt(b.date) || 0;
                return yearB - yearA;
            })
            .slice(0, limit);
    }

    /**
     * Валидирует данные проекта
     */
    validateProjectData(projectData) {
        const errors = [];
        
        if (!projectData.id) {
            errors.push('Project ID is required');
        }
        
        if (!projectData.title) {
            errors.push('Project title is required');
        }
        
        if (!projectData.description) {
            errors.push('Project description is required');
        }
        
        if (projectData.tags && !Array.isArray(projectData.tags)) {
            errors.push('Tags must be an array');
        }

        if (projectData.technologies && !Array.isArray(projectData.technologies)) {
            errors.push('Technologies must be an array');
        }
        
        return {
            isValid: errors.length === 0,
            errors
        };
    }

    /**
     * Получает рекомендации на основе проекта
     */
    async getRecommendations(projectId, limit = 3) {
        const { project } = await this.getProjectData(projectId);
        const allProjects = await this.repository.getAllProjects();
        const recommendations = [];
        
        // Находим проекты с похожими тегами или технологиями
        allProjects.forEach(otherProject => {
            // Исключаем текущий проект
            if (otherProject.getId() === projectId) {
                return;
            }
            
            const similarity = this._calculateSimilarity(project, otherProject);
            if (similarity > 0) {
                recommendations.push({
                    project: otherProject,
                    similarity
                });
            }
        });
        
        return recommendations
            .sort((a, b) => b.similarity - a.similarity)
            .slice(0, limit);
    }

    /**
     * Получает проекты по категориям
     */
    async getProjectsByCategory() {
        const projects = await this.repository.getAllProjects();
        const categories = {};
        
        projects.forEach(project => {
            project.tags.forEach(tag => {
                if (!categories[tag]) {
                    categories[tag] = [];
                }
                categories[tag].push(project);
            });
        });
        
        return categories;
    }

    /**
     * Вычисляет score проекта для featured списка
     */
    _calculateProjectScore(project) {
        let score = 0;
        
        // Баллы за ресурсы
        score += (project.notebooks || 0) * 2;
        score += (project.code || 0) * 3;
        score += (project.demo || 0) * 4; // Demo весит больше
        
        // Баллы за сложность
        const difficultyScores = {
            'Beginner': 1,
            'Intermediate': 2, 
            'Advanced': 3
        };
        score += difficultyScores[project.difficulty] || 0;
        
        // Баллы за популярные технологии
        const popularTechs = ['OpenAI', 'Python', 'React', 'FastAPI'];
        project.technologies.forEach(tech => {
            if (popularTechs.includes(tech)) {
                score += 2;
            }
        });
        
        return score;
    }

    /**
     * Вычисляет схожесть между проектами
     */
    _calculateSimilarity(project1, project2) {
        const tags1 = new Set(project1.tags);
        const tags2 = new Set(project2.tags);
        const techs1 = new Set(project1.technologies);
        const techs2 = new Set(project2.technologies);
        
        // Пересечение тегов
        const tagsIntersection = new Set([...tags1].filter(tag => tags2.has(tag)));
        const tagsUnion = new Set([...tags1, ...tags2]);
        
        // Пересечение технологий
        const techsIntersection = new Set([...techs1].filter(tech => techs2.has(tech)));
        const techsUnion = new Set([...techs1, ...techs2]);
        
        // Коэффициент Жаккара для тегов и технологий
        const tagsSimilarity = tagsIntersection.size / tagsUnion.size;
        const techsSimilarity = techsIntersection.size / techsUnion.size;
        
        // Взвешенная схожесть
        return (tagsSimilarity * 0.6 + techsSimilarity * 0.4);
    }
} 