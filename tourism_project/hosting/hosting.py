from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="tourism_project/deployment",   # the local folder containing your app files
    repo_id="ASNaik/tourism-wellness-app",                     # the target Hugging Face Space
    repo_type="space",
    path_in_repo="",
)
print("Deployment files pushed to the Hugging Face Space.")
