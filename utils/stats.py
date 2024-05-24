def print_stats(epoch, step, loss_D1, loss_D2, loss_G):
    print(f"Epoch [{epoch + 1}], Step [{step}], "
          f"D1 Loss: {loss_D1:.4f}, D2 Loss: {loss_D2:.4f}, "
          f"G Loss: {loss_G:.4f}")