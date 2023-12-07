# <div align='center'>Команда №10 - *Десяптиконы*! :robot:</div>

## Хатакон ООО Просепт х Мастерская Яндекс.Практикум

**Задача проекта:** разработка решения, которое частично автоматизирует процесс сопоставления товаров заказчика с размещаемыми товарами на онлайн площадках дилеров.

**Состав команды:**
- Юлия Никифорова - PM (<img src="https://github.com/mike2023-ml/Portfolio/assets/116313032/d3f08c03-7dec-490e-ad39-75152295c4d5" title="Telegram" alt="Telegram" width="20" height="20"/>@Niki_for_Ova)
- Валентина Ковалëва - DS (<img src="https://github.com/mike2023-ml/Portfolio/assets/116313032/d3f08c03-7dec-490e-ad39-75152295c4d5" title="Telegram" alt="Telegram" width="20" height="20"/>@BrianKowalski)
- Артём Гришин - DS (лид) (<img src="https://github.com/mike2023-ml/Portfolio/assets/116313032/d3f08c03-7dec-490e-ad39-75152295c4d5" title="Telegram" alt="Telegram" width="20" height="20"/>@Owu213)
- Лидия Пономаренко - DS (<img src="https://github.com/mike2023-ml/Portfolio/assets/116313032/d3f08c03-7dec-490e-ad39-75152295c4d5" title="Telegram" alt="Telegram" width="20" height="20"/>@L1d11aP)
- Борис Коренбляс - Backend (лид) (<img src="https://github.com/mike2023-ml/Portfolio/assets/116313032/d3f08c03-7dec-490e-ad39-75152295c4d5" title="Telegram" alt="Telegram" width="20" height="20"/>@bbobr2072)
- Евгений Хомутов - Backend (<img src="https://github.com/mike2023-ml/Portfolio/assets/116313032/d3f08c03-7dec-490e-ad39-75152295c4d5" title="Telegram" alt="Telegram" width="20" height="20"/>@Ev93n1)
- Артём Тулупов - Frontend (<img src="https://github.com/mike2023-ml/Portfolio/assets/116313032/d3f08c03-7dec-490e-ad39-75152295c4d5" title="Telegram" alt="Telegram" width="20" height="20"/>@mrFrontendDev)
- Лия Высоцкая - Frontend (лид) (<img src="https://github.com/mike2023-ml/Portfolio/assets/116313032/d3f08c03-7dec-490e-ad39-75152295c4d5" title="Telegram" alt="Telegram" width="20" height="20"/>@liya_vysockaya)

### Описание проекта находится: [`ipynb`](https://github.com/Decepticons-Hackathon/data-science/blob/main/project_description/hackathon_prosept.ipynb)

**Файлы для взаимодействия с Backend:**
- py-скрипт: [`script`](https://github.com/Decepticons-Hackathon/data-science/blob/main/func_for_back.py)
- обученная модель: [`model`](https://github.com/Decepticons-Hackathon/data-science/blob/main/script/pickle_model.pkl)

## Задачи в DS:
- Анализ данных (все);
- Предобработка текста (Лидия Пономаренко) и категориальных признаков (все);
- Выбор и создание признаков (все);
- Создание таргета (Артем Гришин) и фича-инжиниринг (Валентина Ковалёва);
- Эмбеддинг (все);
- Создание выборки и тестирование модели (Валентина Ковалёва);
- Выгрузка модели (Лидия Пономаренко);
- Написание py-скриптов (Артем Гришин).

**Была выбрана метрика *Presicion*. Достигнуто значение *0.98*.**

### 🧩 Этапы работы:

|    | Гипотеза       | Описание           | Инструменты | Статус    |
|:--:|:--------------:|:------------------:|:-----------:|:---------:|
|1| Наименования товаров можно сопоставить, подсчитав расстояние Левенштейна| Использовали библиотеку для подсчета редакционного расстояния, объем данных для сравнения не большой| fuzzywuzzy|Завершен|
|2| Наименования товаров можно сопоставить, используя embeddings| Использовали предобученный маленький Берт для русского языка, посчитали скалярное произведение векторов, чтобы оценить насколько они сонаправлены| RuBERT tiny|Завершен|
|3| Наименования товаров можно сопоставить с помощью unsupervised learning|||В процессе|

### 📊 Общий вывод:
Для решения задачи заказчика была выбрана рекомендательная модель **LGBMClassifier**. Удалось достигнуть высокой точности предсказаний - 0.98. Тестирование других моделей не потребовалось. Гиперпараметры и валидацию модели провели с GridSearchCV. На этапе изучения данных была обнаружена некорректная запись наименований товаров, отсутствовали пробелы между некоторыми словами, решено предобработкой текста.
