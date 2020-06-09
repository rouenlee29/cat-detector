https://github.com/mohshawky5193/dog-breed-classifier/blob/master/dog-breed-classifier.py
deployed at https://dog-breed-classifier-udacity.herokuapp.com/

from fastai.basic_train import load_learner
from fastai.vision import open_image
import torch
from PIL import Image 


async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()


app = Starlette()

@app.route('/')
async def homepage(request):
    template = "index.html"
    context = {"request": request}
    return templates.TemplateResponse(template, context)

# for user to upload image 
@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    bytes = await (data["file"].read())
    return predict_image_from_bytes(bytes)

# @app.route("/classify-url", methods=["GET"])
# async def classify_url(request):
#     bytes = await get_bytes(request.query_params["url"])
#     return predict_image_from_bytes(bytes)


def predict_image_from_bytes(bytes):
    img = open_image(BytesIO(bytes))

    # load model in export.pkl 
    learn = load_learner(path = ".")

    pred_class,pred_idx,outputs = learn.predict(img)
    i = pred_idx.item()
    classes = ['Domestic Medium Hair', 'Persian', 'Ragdoll', 'Siamese', 'Snowshoe']
    prediction = classes[i]
    
    return JSONResponse({
        "predictions": prediction
    })


@app.route("/")
def form(request):
    return HTMLResponse(
        """
        <form action="/upload" method="post" enctype="multipart/form-data">
            Select image to upload:
            <input type="file" name="file">
            <input type="submit" value="Upload Image">
        </form>
        Or submit a URL:
        <form action="/classify-url" method="get">
            <input type="url" name="url">
            <input type="submit" value="Fetch and analyze image">
        </form>
    """)


@app.route("/form")
def redirect_to_homepage(request):
    return RedirectResponse("/")


if __name__ == "__main__":
    if "serve" in sys.argv:
        uvicorn.run(app, host="0.0.0.0", port=8008)