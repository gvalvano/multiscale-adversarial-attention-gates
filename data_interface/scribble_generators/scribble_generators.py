import numpy as np
from skimage.morphology import skeletonize, dilation, closing


def generate_skeleton_scribble(mask):
    """ Scribbles are approximated by a skeleton of the image
    :param mask: multi-channel binary mask
    :return: scribbles
    """
    # initialize scribbles as empty array
    scribbles = np.zeros_like(mask)
    n_channels = mask.shape[-1]

    for ch in range(n_channels):
        # extract skeleton from the current channel
        m = np.copy(mask[:, :, ch])
        skl = skeletonize(m)

        # make slightly thicker (but always inside the gt mask)
        skl = closing(skl)
        skl = dilation(skl) * m

        # assign skeleton to return array
        scribbles[..., ch] = skl

    return scribbles


def generate_random_walk_scribble(mask, length_coeff=None):
    """ Scribbles are approximated by a random walk
    :param mask: multi-channel binary mask
    :return: scribbles
    """
    if length_coeff is None:
        length_coeff = [0.10, 0.10, 0.10, 0.10]
    return _per_class_random_walk(mask, length_coeff)


def _per_class_random_walk(mask, length_coeff=None):
    """ Generate smooth (self-avoiding) random walk for each class"""

    if length_coeff is None:
        length_coeff = [0.10, 0.10, 0.10, 0.10]
    assert len(mask.shape) == 3  # 2D + class

    W, H, C = mask.shape
    walk_lengths = [int(length_coeff[i] * np.sum(mask[..., i])) for i in range(C)]

    # initialize 3D mask of random walks
    random_walks = np.zeros_like(mask)

    n_channels = mask.shape[-1]
    for ch in range(n_channels):
        m = mask[:, :, ch]

        # get position of pixels belonging to the mask
        where = np.argwhere(m)

        if walk_lengths[ch] >= len(where):
            random_walks[..., ch] = m
        else:
            # initialize 2D mask of zeros to walk in
            _m = np.zeros_like(m)

            # get random seed and initialize to 1
            seed_x, seed_y = where[np.random.randint(0, len(where))]
            _m[seed_x, seed_y] = 1

            last_x, last_y = None, None
            max_iters = 4000
            while len(np.argwhere(_m)) < walk_lengths[ch] and max_iters > 0:
                max_iters -= 1
                x = seed_x + np.random.choice([1, 0, -1])
                y = seed_y + np.random.choice([1, 0, -1])

                if x < 0: x = 0
                if y < 0: y = 0
                if x > W - 1: x = W - 1
                if y > W - 1: y = H - 1

                if not (last_x is None and last_y is None):
                    if last_x != x and last_y != y:
                        # If we are inside the mask, assign 1 to the random walk
                        if m[x, y] == 1:
                            _m[x, y] = 1
                            last_x, last_y = seed_x, seed_y
                            seed_x, seed_y = x, y
                else:
                    # If we are inside the mask, assign 1 to the random walk
                    if m[x, y] == 1:
                        _m[x, y] = 1
                        last_x, last_y = seed_x, seed_y
                        seed_x, seed_y = x, y

            # smooth the random walk:
            _m = closing(dilation(_m))
            _m = skeletonize(_m)

            # make thick (always inside the gt mask)
            _m = dilation(_m) * m

            # assign random walk to the current channel
            random_walks[..., ch] = _m

    return random_walks
