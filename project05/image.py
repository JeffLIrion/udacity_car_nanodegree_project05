import matplotlib.pyplot as plt


def show_images(*args, fontsize=60, show_axes=True):
    """Display 1 or more images
    
    Parameters
    ----------
    *args : list
        A list of ``(img, title)`` pairs OR ``(img, title, outpath)`` triples
    fontsize : int
        The font size for the image titles
    
    """
    n = len(args)
    cols = min((n, ))
    rows = n // cols + (n % cols != 0)

    plt.figure(1, figsize=(48,48))
    for i, ito in enumerate(args):
        plt.subplot(rows, cols, i+1)
        
        if ito[1] is not None:
            plt.title(ito[1], fontsize=fontsize)
        
        if len(ito[0].shape) == 2:
            plt.imshow(ito[0], cmap='hot')
        else:
            plt.imshow(ito[0][:,:,::-1])
        if not show_axes:
            plt.axis('off')
            
        if len(ito) == 3:
            if len(ito[0].shape) == 2:
                plt.imsave(ito[2], ito[0], cmap='hot')
            else:
                plt.imsave(ito[2], ito[0][:,:,::-1])
        
    plt.tight_layout(pad=0., w_pad=0., h_pad=1.0)
    plt.show()
