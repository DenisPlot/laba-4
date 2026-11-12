from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pickle
import numpy as np

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Загружаем модель
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Названия классов (в датасете Ирисов они идут по порядку 0, 1, 2)
target_names = ['setosa', 'versicolor', 'virginica']


@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request,
                  sepal_length: float = Form(...),
                  sepal_width: float = Form(...),
                  petal_length: float = Form(...),
                  petal_width: float = Form(...)):
    # Формируем данные для модели
    data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Предсказываем
    prediction = model.predict(data)[0]
    result_name = target_names[prediction]

    # Возвращаем результат на страницу
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"result": result_name}
    )