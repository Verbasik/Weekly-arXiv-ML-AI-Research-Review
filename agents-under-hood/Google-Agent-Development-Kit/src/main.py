"""
Основная точка входа для Git Workflow Agent

Запускает веб-сервер для взаимодействия с агентом через API.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger
from dotenv import load_dotenv
from contextlib import asynccontextmanager

# Загружаем переменные окружения из .env файла
load_dotenv()

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent))

from git_workflow_agent.agent import create_git_workflow_agent, GitWorkflowAgent
from git_workflow_agent.config import WEB_CONFIG, LOGGING_CONFIG


# Настройка логирования
logger.add(
    LOGGING_CONFIG["file"],
    level=LOGGING_CONFIG["level"],
    format=LOGGING_CONFIG["format"],
    rotation=LOGGING_CONFIG["rotation"],
    retention=LOGGING_CONFIG["retention"]
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan контекст для управления жизненным циклом приложения"""
    # Startup
    logger.info("Git Workflow Agent API запущен")
    logger.info(f"Веб-интерфейс доступен по адресу: http://{WEB_CONFIG['host']}:{WEB_CONFIG['port']}")
    
    yield
    
    # Shutdown
    logger.info("Очистка ресурсов...")
    for agent in agents.values():
        try:
            await agent.cleanup()
        except Exception as e:
            logger.error(f"Ошибка при очистке агента: {e}")
    logger.info("Git Workflow Agent API остановлен")

# Создание FastAPI приложения
app = FastAPI(
    title="Git Workflow Agent API",
    description="API для автоматического анализа изменений и создания коммитов",
    version="1.0.0",
    lifespan=lifespan
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=WEB_CONFIG["cors_origins"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальное хранилище агентов
agents: Dict[str, GitWorkflowAgent] = {}


class ProjectRequest(BaseModel):
    """Модель запроса для работы с проектом"""
    project_path: str


class CommitRequest(BaseModel):
    """Модель запроса для создания коммита"""
    project_path: str
    message: str
    auto_push: bool = False


class AnalysisRequest(BaseModel):
    """Модель запроса для анализа изменений"""
    project_path: str





async def get_or_create_agent(project_path: str) -> GitWorkflowAgent:
    """Получает существующего агента или создает нового"""
    if project_path not in agents:
        logger.info(f"Создание нового агента для проекта: {project_path}")
        agents[project_path] = await create_git_workflow_agent(project_path)
    return agents[project_path]


@app.get("/")
async def root():
    """Корневой endpoint"""
    return {
        "message": "Git Workflow Agent API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса"""
    return {
        "status": "healthy",
        "agents_count": len(agents),
        "timestamp": asyncio.get_event_loop().time()
    }


@app.post("/api/analyze")
async def analyze_changes(request: AnalysisRequest):
    """Анализирует изменения в проекте"""
    try:
        agent = await get_or_create_agent(request.project_path)
        result = await agent.analyze_project_changes()
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except Exception as e:
        logger.error(f"Ошибка при анализе изменений: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/commit")
async def create_commit(request: CommitRequest):
    """Создает коммит с указанным сообщением"""
    try:
        agent = await get_or_create_agent(request.project_path)
        result = await agent.create_commit(
            request.message, 
            auto_push=request.auto_push
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except Exception as e:
        logger.error(f"Ошибка при создании коммита: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/auto-commit")
async def auto_commit_and_push(request: ProjectRequest):
    """Автоматически анализирует изменения, создает коммит и пушит"""
    try:
        agent = await get_or_create_agent(request.project_path)
        result = await agent.auto_commit_and_push()
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except Exception as e:
        logger.error(f"Ошибка при автоматическом коммите: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/status/{project_path:path}")
async def get_project_status(project_path: str):
    """Получает статус проекта"""
    try:
        agent = await get_or_create_agent(project_path)
        result = await agent.get_project_status()
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except Exception as e:
        logger.error(f"Ошибка при получении статуса проекта: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/agent/{project_path:path}")
async def cleanup_agent(project_path: str):
    """Очищает ресурсы агента для указанного проекта"""
    try:
        if project_path in agents:
            await agents[project_path].cleanup()
            del agents[project_path]
            logger.info(f"Агент для проекта {project_path} очищен")
        
        return {"message": f"Агент для проекта {project_path} очищен"}
        
    except Exception as e:
        logger.error(f"Ошибка при очистке агента: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/agents")
async def list_agents():
    """Возвращает список активных агентов"""
    return {
        "agents": list(agents.keys()),
        "count": len(agents)
    }


# CLI интерфейс для тестирования
async def cli_main():
    """CLI интерфейс для тестирования агента"""
    if len(sys.argv) < 2:
        print("Использование: python main.py <project_path> [analyze|commit|auto-commit]")
        sys.exit(1)
    
    project_path = sys.argv[1]
    command = sys.argv[2] if len(sys.argv) > 2 else "analyze"
    
    try:
        agent = await create_git_workflow_agent(project_path)
        
        if command == "analyze":
            result = await agent.analyze_project_changes()
            print("Результат анализа:")
            print(result)
            
        elif command == "commit":
            if len(sys.argv) < 4:
                print("Укажите сообщение коммита")
                sys.exit(1)
            message = sys.argv[3]
            result = await agent.create_commit(message, auto_push=True)
            print("Результат коммита:")
            print(result)
            
        elif command == "auto-commit":
            result = await agent.auto_commit_and_push()
            print("Результат автоматического коммита:")
            print(result)
            
        else:
            print(f"Неизвестная команда: {command}")
            sys.exit(1)
            
        await agent.cleanup()
        
    except Exception as e:
        logger.error(f"Ошибка в CLI: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Проверяем переменные окружения
    if not os.getenv("GOOGLE_API_KEY"):
        logger.error("Не установлена переменная GOOGLE_API_KEY")
        sys.exit(1)
    
    # Запускаем CLI или сервер
    if len(sys.argv) > 1 and sys.argv[1] != "server":
        asyncio.run(cli_main())
    else:
        import uvicorn
        uvicorn.run(
            app,
            host=WEB_CONFIG["host"],
            port=WEB_CONFIG["port"],
            log_level="info"
        ) 