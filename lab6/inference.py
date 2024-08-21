import yaml
import torch
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from diffusers import DDPMScheduler
from dataset import CLEVRDatasetEval
from models.conditional_unet import ClassConditionedUnet
from utils import set_random_seed
from evaluator import evaluation_model


def inference(config, dataloader, model):
    # Noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config["denoise_steps"], beta_schedule="squaredcos_cap_v2"
    )

    # Prepare to store results
    save_timesteps = (
        torch.linspace(0, config["denoise_steps"] - 1, steps=11).long().tolist()
    )  # timesteps to save images
    denoise_images_list = [[] for _ in range(len(eval_dataset))]
    final_results = []

    # Inference loop
    for i, labels in enumerate(dataloader):
        labels = labels.cuda()
        x = torch.randn((1, 3, config["image_size"], config["image_size"])).cuda()

        for t_idx, t in tqdm(
            enumerate(noise_scheduler.timesteps), total=len(noise_scheduler.timesteps)
        ):
            # Predict residual
            with torch.no_grad():
                residual = model(x, t, labels)

            # Update sample
            x = noise_scheduler.step(residual, t, x).prev_sample

            if t_idx in save_timesteps:
                denoise_images_list[i].append(x.clone().cpu().squeeze())

        final_results.append(x.clone().cpu().squeeze())

    return denoise_images_list, final_results


if __name__ == "__main__":
    # Load arguments from config file
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Set random seed
    set_random_seed(config["seed"])

    # Load the evaluation dataset
    eval_dataset = CLEVRDatasetEval(
        eval_json=config["eval_json_path"],
        objects_json=config["objects_json_path"],
    )
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    # Load the model
    model = ClassConditionedUnet(num_classes=config["num_classes"])
    model.load_state_dict(torch.load(config["model_path"], weights_only=True))
    model = model.cuda()
    model.eval()

    denoise_images_list, final_results = inference(config, eval_dataloader, model)

    # Save results
    directories = ["", "denoise", "synthesized"]
    for dir_name in directories:
        os.makedirs(os.path.join(config["result_dir"], dir_name), exist_ok=True)

    # Save denoising results
    for i, denoise_images in enumerate(denoise_images_list):
        denoise_images = torch.stack(denoise_images)
        result_path = os.path.join(config["result_dir"], "denoise", f"label_{i}.png")
        grid = vutils.make_grid(denoise_images, nrow=11, normalize=True, padding=2)
        vutils.save_image(grid, result_path)

    # Save synthesized results
    for i, final_result in enumerate(final_results):
        result_path = os.path.join(
            config["result_dir"], "synthesized", f"label_{i}.png"
        )
        vutils.save_image(final_result, result_path, normalize=True)

    # Save test results grid
    result_path = os.path.join(config["result_dir"], "test_results.png")
    final_results_tensor = torch.stack(final_results)
    grid = vutils.make_grid(final_results_tensor, nrow=8, normalize=True, padding=2)
    vutils.save_image(grid, result_path)

    # Evaluate the results
    evaluator = evaluation_model()
    final_results_tensor = final_results_tensor.cuda()
    labels = eval_dataset.labels.cuda()
    accuracy = evaluator.eval(final_results_tensor, labels)
    print(
        f"Classification accuracy: {accuracy}, testing data: {config["eval_json_path"]}"
    )
