import wandb

def init_wandb(project_name, config, run_name=None, entity=None, tags=None):
    run = wandb.init(
        project=project_name,
        entity=entity,
        name=run_name,
        config=config,
        tags=tags or []
    )
    return run

def log_training_metrics(loss, current_lr, epoch, iteration, reco_loss=None):
    if wandb.run is None:
        return
    
    metrics = {
        "train/loss": loss,
        "train/lr": current_lr,
        "train/epoch": epoch,
        "train/iteration": iteration,
    }
    
    if reco_loss is not None:
        metrics["train/reco_loss"] = reco_loss
    
    wandb.log(metrics, step=iteration)

def log_validation_metrics(val_loss, mean_iou, iteration, class_ious=None, images=None):
    if wandb.run is None:
        return
    
    metrics = {
        "val/loss": val_loss,
        "val/mean_iou": mean_iou,
    }
    
    if class_ious is not None:
        for c, iou in enumerate(class_ious):
            metrics[f"val/class_iou_{c}"] = iou
    
    if images is not None:
        metrics["val/predictions"] = images
    
    wandb.log(metrics, step=iteration)

def watch_model(model):
    if wandb.run is None:
        return
    
    wandb.watch(model, log="all", log_freq=100)

def update_summary(best_iou, iteration):
    if wandb.run is None:
        return
    
    wandb.run.summary["best_iou"] = best_iou
    wandb.run.summary["best_model_iteration"] = iteration

def finish():
    if wandb.run is None:
        return
    
    wandb.finish()