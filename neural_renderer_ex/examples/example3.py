"""
Example 3. Optimizing textures.
"""
from __future__ import division
import os
import argparse
import glob

import torch
import torch.nn as nn
import numpy as np
from skimage.io import imread, imsave
import tqdm
import imageio
import scipy.misc
import neural_renderer as nr
from scipy import ndimage


current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')

class Model(nn.Module):
    def __init__(self, filename_obj, filename_ref):
        super(Model, self).__init__()
        vertices, faces = nr.load_obj(filename_obj)
        self.register_buffer('vertices', vertices[None, :, :])
        self.register_buffer('faces', faces[None, :, :])

        # create textures
        texture_size = 4
        textures = torch.zeros(1, self.faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32)
        self.textures = nn.Parameter(textures)

        # load reference image
        # image_ref = torch.from_numpy(imread(filename_ref).astype('float32') / 255.).permute(2,0,1)[None, ::]
        filename_ref = scipy.misc.imread(filename_ref).astype('float32')
        filename_ref = scipy.misc.imresize(filename_ref, (1024, 1024))
        filename_ref = ndimage.rotate(filename_ref, 270)
        image_ref = torch.from_numpy(filename_ref.astype('float32') / 255.).permute(2,0,1)[None, ::]

        self.register_buffer('image_ref', image_ref)

        # setup renderer
        renderer = nr.Renderer(image_size=1024,camera_mode='look_at')
        renderer.perspective = False
        renderer.light_intensity_directional = 0.0
        renderer.light_intensity_ambient = 1.0
        self.renderer = renderer


    def forward(self):
        self.renderer.eye = nr.get_points_from_angles(2.732, 0, np.random.uniform(0, 360))
        image, _, _ = self.renderer(self.vertices, self.faces, torch.tanh(self.textures))
        loss = torch.sum((image - self.image_ref) ** 2)
        return loss


def make_gif(filename):
    with imageio.get_writer(filename, mode='I') as writer:
        for filename in sorted(glob.glob('/tmp/_tmp_*.png')):
            writer.append_data(imageio.imread(filename))
            os.remove(filename)
    writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-io', '--filename_obj', type=str, default=os.path.join(data_dir, '/home/cai/Downloads/deep_dream_3d/examples/data/Laurana50k.obj'))
    # parser.add_argument('-io', '--filename_obj', type=str, default=os.path.join(data_dir, '/home/cai/Downloads/DECA/data/head_template.obj'))
    parser.add_argument('-ir', '--filename_ref', type=str, default=os.path.join(data_dir, '/home/cai/Downloads/style_transfer_3d/examples/data/styles/maxy1.jpg'))
    parser.add_argument('-or', '--filename_output', type=str, default=os.path.join(data_dir, '/home/cai/Downloads/face_style_transfer/maxy1_face.gif'))
    parser.add_argument('-ob', '--object_output', type=str,
                        default=os.path.join(data_dir, '/home/cai/Downloads/face_style_transfer/maxy1.obj'))
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()

    model = Model(args.filename_obj, args.filename_ref)
    model.cuda()
    #0.1 50
    # 25 imagesize 1024

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, betas=(0.5,0.999))
    loop = tqdm.tqdm(range(25))
    for i in loop:
        loop.set_description('Optimizing')
        optimizer.zero_grad()
        loss = model()
        # if loss.item() < 950000 and i > 10 :
        #     break
        print(loss)
        loss.backward()
        optimizer.step()


    model.faces = torch.cat((model.faces, model.faces[:, :, list(reversed(range(model.faces.shape[-1])))]), dim=1).detach()
    textures = torch.cat((model.textures, model.textures.permute((0, 1, 4, 3, 2, 5))), dim=1)
    faces_lighting = nr.vertices_to_faces( model.vertices, model.faces)
    textures = nr.lighting(faces_lighting,textures,1,0,[1,1,1],[1,1,1],[0,1,0])

    textures = torch.cat((model.textures, model.textures.permute((0, 1, 4, 3, 2, 5))), dim=1)
    nr.save_obj(args.object_output, model.vertices[0], model.faces[0], torch.tanh(textures[0]))

    # nr.save_obj(args.object_output, model.vertices[0], model.faces[0], torch.tanh(model.textures[0]))
    # draw object
    # loop = tqdm.tqdm(range(0, 360, 4))
    # for num, azimuth in enumerate(loop):
    #     loop.set_description('Drawing')
    #     model.renderer.eye = nr.get_points_from_angles(2.732, 60, azimuth)
    #     images, _, _ = model.renderer(model.vertices, model.faces, torch.tanh(model.textures))
    #     image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))
    #     imsave('/tmp/_tmp_%04d.png' % num, image)
    # make_gif(args.filename_output)


if __name__ == '__main__':
    main()
