# A2A/hosts/cli/__main__.py
"""
Модуль реализует CLI-клиент для взаимодействия с A2A (Agent-to-Agent) API.

Позволяет отправлять сообщения агенту, получать ответы в потоковом или обычном режиме,
а также опционально использовать push-уведомления для получения обновлений от агента.
"""

# Стандартные библиотеки
import asyncio
import urllib.parse
from uuid import uuid4

# Сторонние библиотеки
import asyncclick as click

# Внутренние модули
from common.client import A2AClient, A2ACardResolver
from common.types import TaskState, Task
from common.utils.push_notification_auth import PushNotificationReceiverAuth


@click.command()
@click.option("--agent", default="http://localhost:10000", help="URL агента для взаимодействия")
@click.option("--session", default=0, help="ID сессии (0 = создать новую)")
@click.option("--history", default=False, help="Показывать историю сообщений")
@click.option("--use_push_notifications", default=False, help="Использовать push-уведомления")
@click.option(
    "--push_notification_receiver", 
    default="http://localhost:5000", 
    help="URL для получения push-уведомлений"
)
async def cli(
    agent: str, 
    session: int, 
    history: bool, 
    use_push_notifications: bool, 
    push_notification_receiver: str
) -> None:
    """
    Description:
    ---------------
        Точка входа CLI-клиента для взаимодействия с A2A агентом.
        
    Args:
    ---------------
        agent: URL агента для взаимодействия
        session: ID сессии (0 = создать новую)
        history: Показывать историю сообщений
        use_push_notifications: Использовать push-уведомления
        push_notification_receiver: URL для получения push-уведомлений
        
    Returns:
    ---------------
        None
        
    Examples:
    ---------------
        >>> asyncio.run(cli("http://localhost:10000", 0, False, False, "http://localhost:5000"))
    """
    # Получаем карточку агента
    card_resolver = A2ACardResolver(agent)
    card = card_resolver.get_agent_card()

    print("======= Agent Card ========")
    print(card.model_dump_json(exclude_none=True))

    # Парсим URL для push-уведомлений
    notif_receiver_parsed = urllib.parse.urlparse(push_notification_receiver)
    notification_receiver_host = notif_receiver_parsed.hostname
    notification_receiver_port = notif_receiver_parsed.port

    # Настраиваем прослушивание push-уведомлений, если нужно
    if use_push_notifications:
        from hosts.cli.push_notification_listener import PushNotificationListener
        notification_receiver_auth = PushNotificationReceiverAuth()
        await notification_receiver_auth.load_jwks(f"{agent}/.well-known/jwks.json")

        push_notification_listener = PushNotificationListener(
            host=notification_receiver_host,
            port=notification_receiver_port,
            notification_receiver_auth=notification_receiver_auth,
        )
        push_notification_listener.start()
        
    # Инициализируем клиент
    client = A2AClient(agent_card=card)
    
    # Создаем или используем существующий ID сессии
    if session == 0:
        session_id = uuid4().hex
    else:
        session_id = str(session)

    continue_loop = True
    streaming = card.capabilities.streaming

    # Основной цикл взаимодействия с агентом
    while continue_loop:
        task_id = uuid4().hex
        print("=========  starting a new task ======== ")
        continue_loop = await complete_task(
            client, 
            streaming, 
            use_push_notifications, 
            notification_receiver_host, 
            notification_receiver_port, 
            task_id, 
            session_id
        )

        # Показываем историю, если это требуется
        if history and continue_loop:
            print("========= history ======== ")
            task_response = await client.get_task({"id": task_id, "historyLength": 10})
            print(task_response.model_dump_json(include={"result": {"history": True}}))


# Функция для обработки одного задания (task)
async def complete_task(
    client: A2AClient, 
    streaming: bool, 
    use_push_notifications: bool, 
    notification_receiver_host: str, 
    notification_receiver_port: int, 
    task_id: str, 
    session_id: str
) -> bool:
    """
    Description:
    ---------------
        Обрабатывает одно задание для агента, отправляя запрос и обрабатывая ответ.
        
    Args:
    ---------------
        client: Клиент A2A API
        streaming: Флаг, использовать ли потоковую передачу данных
        use_push_notifications: Использовать ли push-уведомления
        notification_receiver_host: Хост для получения push-уведомлений
        notification_receiver_port: Порт для получения push-уведомлений
        task_id: Идентификатор задания
        session_id: Идентификатор сессии
        
    Returns:
    ---------------
        bool: True, если следует продолжить цикл взаимодействия, иначе False
        
    Raises:
    ---------------
        Exception: При ошибке отправки запроса или обработки ответа
        
    Examples:
    ---------------
        >>> await complete_task(client, True, False, "localhost", 5000, "task123", "session456")
        True
    """
    # Запрашиваем ввод пользователя
    prompt = click.prompt(
        "\nWhat do you want to send to the agent? (:q or quit to exit)"
    )
    
    # Проверяем команду выхода
    if prompt in [":q", "quit"]:
        return False

    # Формируем запрос
    payload = {
        "id": task_id,
        "sessionId": session_id,
        "acceptedOutputModes": ["text"],
        "message": {
            "role": "user",
            "parts": [
                {
                    "type": "text",
                    "text": prompt,
                }
            ],
        },
    }

    # Добавляем информацию о push-уведомлениях, если нужно
    if use_push_notifications:
        payload["pushNotification"] = {
            "url": f"http://{notification_receiver_host}:{notification_receiver_port}/notify",
            "authentication": {
                "schemes": ["bearer"],
            },
        }

    task_result = None
    
    # Отправляем запрос в зависимости от режима (потоковый или обычный)
    if streaming:
        response_stream = client.send_task_streaming(payload)
        async for result in response_stream:
            print(f"stream event => {result.model_dump_json(exclude_none=True)}")
        task_result = await client.get_task({"id": task_id})
    else:
        task_result = await client.send_task(payload)
        print(f"\n{task_result.model_dump_json(exclude_none=True)}")

    # Проверяем, требуется ли дополнительный ввод
    state = TaskState(task_result.result.status.state)
    if state.name == TaskState.INPUT_REQUIRED.name:
        # Рекурсивно запрашиваем дополнительный ввод
        return await complete_task(
            client,
            streaming,
            use_push_notifications,
            notification_receiver_host,
            notification_receiver_port,
            task_id,
            session_id
        )
    else:
        # Задание завершено
        return True


if __name__ == "__main__":
    asyncio.run(cli())