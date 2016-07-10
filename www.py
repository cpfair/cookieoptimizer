from flask import Flask, render_template, request, redirect, make_response
from optimizer import capture_shapes, capture_container, pack_shapes
from collections import defaultdict
import uuid
import threading
import random

# We can't stuff all the data in Flask's sessions.
sess_data = defaultdict(dict)
job_data = defaultdict(dict)

app = Flask(__name__)
app.secret_key = "tasty"

@app.route('/upload_shapes', methods=['GET', 'POST'])
def upload_shapes():
    sess_dict = sess_data[request.cookies["id"]]
    if request.method == 'POST':
        file = request.files['file']
        file.save("data/shapes.jpg")
        shapes, shape_images = capture_shapes("data/shapes.jpg", "static")
        sess_dict["shapes"] = shapes
        sess_dict["shape_images"] = shape_images
        sess_dict["job"] = None
        return ""
    return render_template("upload.html", title="Upload Cookie Cutters", target="/upload_shapes")

@app.route('/upload_container', methods=['GET', 'POST'])
def upload_container():
    sess_dict = sess_data[request.cookies["id"]]
    if request.method == 'POST':
        file = request.files['file']
        file.save("data/container.jpg")
        base_image, container = capture_container("data/container.jpg")
        sess_dict["container"] = container
        sess_dict["base_image"] = base_image
        sess_dict["job"] = None
        return ""
    return render_template("upload.html", title="Upload Cookie Dough", target="/upload_container", shape_images=sess_dict["shape_images"], random=random.random())

def run_job(job_id, result_image, container, shapes, result_img_path):
    def cb(cookies, attempts):
        job_data[job_id]["cookies"] = cookies
        job_data[job_id]["attempts"] = attempts
    pack_shapes(result_image, container, shapes, result_img_path, cb)
    job_data[job_id]["done"] = True

@app.route('/')
def result():
    if "id" not in request.cookies:
        session_id = str(uuid.uuid4())
    else:
        session_id = request.cookies["id"]
    sess_dict = sess_data[session_id]
    if "shapes" not in sess_dict:
        resp = make_response(redirect("upload_shapes"))
        # I changed this from using sessions, then realized that made little sense.
        # Too late now.
        resp.set_cookie("id", session_id)
        return resp
    if "container" not in sess_dict:
        return redirect("upload_container")

    if not sess_dict["job"]:
        job_id = str(uuid.uuid4())
        threading.Thread(target=run_job, args=(job_id, sess_dict["base_image"], sess_dict["container"], sess_dict["shapes"], "static/result.jpg")).start()
        sess_dict["job"] = job_id

    if not job_data[sess_dict["job"]].get("done", False):
        return render_template("wait.html", cookies=job_data[sess_dict["job"]].get("cookies", 0), attempts=job_data[sess_dict["job"]].get("attempts", 0))
    return render_template("result.html", random=random.random())

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
