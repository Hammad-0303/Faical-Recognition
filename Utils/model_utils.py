import torch
def save_model(model, optimizer,epoch,loss, path = "face_embedding_model.pth"):
    # Saves the model's state, optimizer state, and training metadata.
    
    # Parameters:
    #     model (nn.Module): The trained model.
    #     optimizer (torch.optim.Optimizer): The optimizer used during training.
    #     epoch (int): Current epoch number.
    #     loss (float): Training loss at the time of saving.
    #     path (str): File path to save the model.
    checkpoint = {'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'loss': loss}
    torch.save(checkpoint, path)
    print(f"Model saved at {path} during epoch {epoch} with loss {loss:.4f}")


def load_model(model, optimizer, path = "face_embedding_model.pth"):
    # Loads the model's state, optimizer state, and training metadata.
    
    # Parameters:
    #     model (nn.Module): The model to load the state_dict into.
    #     optimizer (torch.optim.Optimizer): The optimizer to load the state_dict into.
    #     path (str): File path to load the model.
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer, checkpoint['epoch'], checkpoint['loss']
   