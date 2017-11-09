import h5py
from keras.engine.topology import Layer
from keras import backend as K
import numpy as np


class Standardize(Layer):
    def __init__(self, std_stats, take_particles=True, take_HLF=False, **kwargs):
        self.std_stats = std_stats
        self.take_particles = take_particles
        self.take_HLF = take_HLF
        with h5py.File(std_stats) as f:
            # print(f.keys())
            self.particle_mean = np.array(f['particle_mean'][:])
            self.particle_std = np.array(f['particle_std'][:])
            self.hlf_mean = np.array(f['hlf_mean'][:])
            self.hlf_std = np.array(f['hlf_std'][:])
        # print(self.particle_mean)
        # print(self.particle_std)
        self.particle_mean = self.particle_mean.reshape((1, self.particle_mean.shape[-1]))
        self.particle_std = self.particle_std.reshape((1, self.particle_std.shape[-1]))
        # self.hlf_mean = self.hlf_mean.reshape(1, self.hlf_mean[-1])
        # self.hlf_std = self.hlf_std.reshape(1, self.hlf_std[-1])
        super(Standardize, self).__init__(**kwargs)

    def build(self, input_shapes):
        l = 1 if isinstance(input_shapes, tuple) else len(input_shapes)
        nb_inps = int(self.take_particles) + int(self.take_HLF)
        assert l == nb_inps, "%r != %r" % (l, nb_inps)
        self.built = True

    def compute_mask(self, input, input_mask=None):
        return input_mask  # [None] * (int(self.take_particles) + int(self.take_HLF))

    def call(self, inp, mask=None):
        # nb_inps =int(self.take_particles) + int(self.take_HLF)
        # assert inp.shape[0] == nb_inps, "%r != %r" % (inp.shape[0],nb_inps) 
        if (self.take_particles):
            if (self.take_HLF):
                particles, hlf = inp
            else:
                particles = inp
        elif (self.take_HLF):
            hlf = inp
        else:
            raise ValueError("take_particles=%r, take_HLF=%r, cannot both be False." % (self.take_particles, self.take_HLF))

        out = []
        if (self.take_particles):
            particles = particles - K.cast(self.particle_mean, dtype=particles.dtype)
            particles = particles / K.cast(self.particle_std, dtype=particles.dtype)
            out.append(particles)
        if (self.take_HLF):
            hlf = hlf - K.cast(self.hlf_mean, dtype=hlf.dtype)
            hlf = hlf / K.cast(self.hlf_std, dtype=hlf.dtype)
            out.append(hlf)

        if (len(out) == 1): out = out[0]
        return out

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_config(self):
        base_config = Layer.get_config(self)
        config = {'std_stats': self.std_stats,
                  'take_particles': self.take_particles,
                  'take_HLF': self.take_HLF}
        return dict(list(base_config.items()) + list(config.items()))