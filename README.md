### Trained Model Weights: [Hugging Face](https://huggingface.co/ТВОЙ_НИК/unique-code-rl)
### Base Model: [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)

## A project for training large language models (LLMs) to generate structurally unique code using reinforcement learning (GRPO).


Features:
Reward and penalty system: The model receives a base score for passing tests, a bonus for a new code structure, and a penalty for copying the structures of other responses in the batch. This motivates the model to search for alternative algorithms instead of writing the same template.


Custom Evaluator (evaluator.py):
A robust code verification system using AST, fuzz-testing, and automatic removal of cosmetic junk. It protects the system from hardcoding and attempts by the model to cheat the reward.


Memory efficiency:
Training was conducted on a consumer GPU (RTX 5070 Laptop with 8 GB VRAM). Used the Qwen3-4B model (4 billion parameters) with 4-bit quantization.


Results and "Aha Moment":
After 1000 training steps, the model successfully overcame the tendency to repeat the same code. The average number of unique code structures per task increased from 1.0 to 1.4. On 33% of tasks, the model generated 2 to 5 structurally different solutions. In addition, the model began to demonstrate conscious optimization.

Example from the logs:
Solving a task to find all divisors of an integer, the model initially considered a simple approach: checking every number from 1 to n. However, during its thinking process, it stopped and wrote: "Wait, but for large n this might be inefficient... I can optimize this by checking only up to the square root of n". Then it proactively used math.isqrt(n) to avoid floating-point issues, and successfully optimized the algorithm on its own, without any specific instructions.


Installation and Launch:

1. Install WSL2 (Ubuntu) and Python 3.11.

2. Create a virtual environment and activate it:
python3.11 -m venv .venv
source .venv/bin/activate

3. Install PyTorch with CUDA 13.0 support (this version is required for the RTX 50 series):
pip install --pre torch==2.9.1+cu130
torchvision==0.24.1+cu130
torchaudio==2.9.1+cu130 --index-url
https://download.pytorch.org/whl/cu130

Note: If you have a different GPU series (e.g., RTX 30 or 40), install PyTorch with the appropriate CUDA version (e.g., cu121 or cu118). You can find the commands for your version on the official PyTorch website.

4. Check that PyTorch sees the GPU:
python -c "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
It should output True and the name of your GPU.

5. Install the remaining libraries:
pip install unsloth trl peft accelerate bitsandbytes datasets transformers

6. Download the base model:
hf download Qwen/Qwen3-4B --local-dir./Qwen3-4B

7. Ensure that the files evaluator.py, memory_cleanup.py, and train_rl.py are in the same folder as the model.

8. Start training:
python train_rl.py

Note: The mbpp dataset (about 956 tasks) is downloaded automatically on the first run. The number of training steps is set in the NUM_STEPS variable in the train_rl.py file (default is 1000).


### FAQ
### Read the [FAQ](FAQ.md) for technical details.

---

### Веса обученной модели: [Hugging Face](https://huggingface.co/ТВОЙ_НИК/unique-code-rl)
### Базовая модель: [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)

## Проект по обучению больших языковых моделей (LLM) генерации структурно уникального кода с помощью обучения с подкреплением (GRPO).


Особенности:
Система наград и штрафов: Модель получает базовый балл за прохождение тестов, бонус за новую структуру кода и штраф за копирование структур других ответов в пакете. Это мотивирует модель искать альтернативные алгоритмы, а не писать один и тот же шаблон.


Кастомный Evaluator (evaluator.py):
Надежная система проверки кода, использующая AST, fuzz-тестирование и автоматическое удаление косметического мусора. Защищает систему от хардкода и попыток модели обмануть награду.


Эффективность по памяти:
Обучение проводилось на потребительской видеокарте (RTX 5070 Laptop с 8 ГБ VRAM). Использована модель Qwen3-4B (4 миллиарда параметров) с 4-битным квантованием.


Результаты и момент озарения:
После 1000 шагов обучения модель успешно преодолела склонность к повторению одного и того же кода. Среднее число уникальных структур кода на задачу выросло с 1.0 до 1.4. На 33% задач модель сгенерировала от 2 до 5 структурно разных решений. Кроме того, модель начала демонстрировать осознанную оптимизацию.

Пример из логов:
Решая задачу на поиск всех делителей целого числа, модель изначально рассмотрела простой подход: проверять каждое число от 1 до n. Однако в процессе размышлений она остановилась и написала: «Подожди, но для больших n это может быть неэффективно... Я могу оптимизировать это, проверяя только до квадратного корня из n». Затем она по своей инициативе использовала math.isqrt(n), чтобы избежать проблем с плавающей точкой, и успешно оптимизировала алгоритм сама, без каких-либо конкретных указаний на это.


Установка и запуск:

1. Установите WSL2 (Ubuntu) и Python 3.11.

2. Создайте виртуальное окружение и активируйте его:
python3.11 -m venv .venv
source .venv/bin/activate

3. Установите PyTorch с поддержкой CUDA 13.0 (эта версия нужна для RTX 50-й серии):
pip install --pre torch==2.9.1+cu130
torchvision==0.24.1+cu130
torchaudio==2.9.1+cu130 --index-url
https://download.pytorch.org/whl/cu130

Примечание: если у вас видеокарта другой серии (например, RTX 30 или 40), установите PyTorch с подходящей версией CUDA (например, cu121 или cu118). Команды для вашей версии можно найти на официальном сайте PyTorch.

4. Проверьте, что PyTorch видит видеокарту:
python -c "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
Должно вывести True и название вашей видеокарты.

5. Установите остальные библиотеки:
pip install unsloth trl peft accelerate bitsandbytes datasets transformers

6. Скачайте базовую модель:
hf download Qwen/Qwen3-4B --local-dir./Qwen3-4B

7. Убедитесь, что файлы evaluator.py, memory_cleanup.py и train_rl.py лежат в одной папке с моделью.

8. Запустите обучение:
python train_rl.py

Примечание: датасет mbpp (около 956 задач) скачивается автоматически при первом запуске. Количество шагов обучения задается в переменной NUM_STEPS в файле train_rl.py (по умолчанию установлено 1000).


### FAQ
### Прочитать [FAQ](FAQ.md) для получения технических деталей.
