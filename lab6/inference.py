import yaml
import torch
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
from dataset import CLEVRDatasetEval
from models.conditional_unet import ClassConditionedUnet
from utils import set_random_seed, show_denoising_grid, show_test_results_grid


def inference(config):
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

    # Noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2"
    )

    # Prepare to store results
    save_timesteps = torch.linspace(0, 999, steps=11).long().tolist()  # timesteps to save images
    denoise_images_list = [
        [] for _ in range(len(eval_dataset))
    ]
    final_results = []

    # Inference loop
    for i, labels in enumerate(eval_dataloader):
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

    # Save results
    os.makedirs(config["result_dir"], exist_ok=True)

    for i, denoise_images in enumerate(denoise_images_list):
        denoise_images = torch.stack(denoise_images)
        result_path = os.path.join(config["result_dir"], f"label_{i}.png")
        show_denoising_grid(denoise_images, result_path)

    final_results_tensor = torch.stack(final_results)
    show_test_results_grid(final_results_tensor, config["result_dir"])


if __name__ == "__main__":
    # Load arguments from config file
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Set random seed
    set_random_seed(config["seed"])

    inference(config)
