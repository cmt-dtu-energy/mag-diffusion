import os
import sys
from multiprocessing import Process, cpu_count
from pathlib import Path

import h5py
import numpy as np
from magtense.magstatics import Tiles, grid_config, run_simulation
from tqdm import tqdm


def db_magfield(
    datapath: Path,
    n_samples: int,
    res: list[int],
    spots: list = [10, 10, 5],
    area: list = [1, 1, 0.5],
    gap: float = 0.05,
    seed: int = 0,
    intv: list | None = None,
    name: str = "magfield",
    empty: bool = False,
) -> None:
    """
    Generate 3-D magnetic fields of experimental setup.

    Args:
        filepath: Indicates where to store the data sample.
        n_samples: Size of database.
        res: Resolution of magnetic field.
        spots: Available positions in setup.
        area: Size of setup.
        gap: Gap between measurement area and surrounding magnets.
        seed: Seed for random number generator of matrices.
        intv: Data range to iterate over.
        empty: If set, an empty database is created.
    """
    fname = f"{name}_{res[0]}"
    if intv is None:
        intv = [0, n_samples]
    if not empty:
        fname += f"_{intv[0]}_{intv[1]}"
    n_intv = intv[1] - intv[0]

    db = h5py.File(f"{datapath}/{fname}.h5", libver="latest", mode="w")
    out_shape = (n_intv, 3, *res)
    db.create_dataset("field", shape=out_shape, dtype="float32")
    if not empty:
        db.attrs["intv"] = intv

    if empty:
        db.attrs["spots"] = spots
        db.attrs["area"] = area
        db.attrs["gap"] = gap
        db.attrs["seed"] = seed
        db.close()

        return fname, n_samples

    rng = np.random.default_rng(seed)
    tile_size = np.asarray(area) / np.asarray(spots)
    filled_mat = rng.integers(2, size=(n_samples, spots[0], spots[1], spots[2]))
    empty_mat = rng.integers(4, size=(n_samples,))

    for idx in tqdm(range(n_intv)):
        emp_x, emp_y = {0: [4, 5], 1: [4, 6], 2: [3, 6], 3: [3, 7]}[
            empty_mat[idx + intv[0]]
        ]
        s_x = emp_x * tile_size[0]
        s_y = emp_y * tile_size[1]

        filled_pos = [
            [i, j, k]
            for i in range(spots[0])
            for j in range(spots[1])
            for k in range(spots[2])
            if filled_mat[intv[0] + idx][i][j][k] == 1
            and (i < emp_x or i > emp_y or j < emp_x or j > emp_y or k < 2 or k > 2)
        ]

        tiles, _ = grid_config(spots, area, filled_pos)

        x_eval = np.linspace(s_x + gap, s_y + gap, res[0])
        y_eval = np.linspace(s_x + gap, s_y + gap, res[1])

        if len(res) == 2:
            xv, yv = np.meshgrid(x_eval, y_eval)
            zv = np.zeros(res[0] * res[1]) + area[2] / 2

        elif len(res) == 3:
            # Pixel length in z-direction equal to x-direction
            s_z = (s_y - s_x) / res[0]
            z_eval = np.linspace(-s_z, s_z, res[2]) + area[2] / 2
            xv, yv, zv = np.meshgrid(x_eval, y_eval, z_eval)

        else:
            err_msg = "Only 2-D and 3-D magnetic field can be generated!"
            raise ValueError(err_msg)

        pts_eval = np.hstack([xv.reshape(-1, 1), yv.reshape(-1, 1), zv.reshape(-1, 1)])
        devnull = open("/dev/null", "w")  # noqa: PTH123, SIM115
        oldstdout_fno = os.dup(sys.stdout.fileno())
        os.dup2(devnull.fileno(), 1)
        _, h_out = run_simulation(tiles, pts_eval)
        os.dup2(oldstdout_fno, 1)

        # Tensor image with shape CxHxWxD [T]
        field = h_out.reshape((*res, 3))
        if len(res) == 3:
            field = field.transpose((3, 0, 1, 2))
        else:
            field = field.transpose((2, 0, 1))

        # Transpose the field to have 'ij' indexing
        field_ij = np.stack([field[0].T, field[1].T, field[2].T], axis=0)
        db["field"][idx] = field_ij * 4 * np.pi * 1e-7

    db.close()


def db_magfield_symm(  # noqa: PLR0912
    datapath: Path,
    n_samples: int,
    res: list[int],
    spots: list = [10, 10, 5],
    area: list = [1, 1, 0.5],
    gap: float = 0.05,
    seed: int = 0,
    intv: list | None = None,
    name: str = "magfield_symm",
    empty: bool = False,
    z_comp: bool = False,
) -> None:
    """
    Generate 3-D magnetic fields of experimental setup.

    Args:
        filepath: Indicates where to store the data sample.
        n_samples: Size of database.
        res: Resolution of magnetic field.
        spots: Available positions in setup.
        area: Size of setup.
        gap: Gap between measurement area and surrounding magnets.
        seed: Seed for random number generator of matrices.
        intv: Data range to iterate over.
        empty: If set, an empty database is created.
    """
    symm = True
    fname = f"{name}_{res[0]}_{n_samples}"
    if z_comp:
        fname += "_z"
    if intv is None:
        intv = [0, n_samples]
    if not empty:
        fname += f"_{intv[0]}_{intv[1]}"
    n_intv = intv[1] - intv[0]

    db = h5py.File(f"{datapath}/{fname}.h5", libver="latest", mode="w")
    out_shape = (n_intv, 3, *res) if z_comp else (n_intv, 2, *res)
    db.create_dataset("field", shape=out_shape, dtype="float32")
    if not empty:
        db.attrs["intv"] = intv

    if symm:
        spots[2] = 1
        area[2] = 10

    if empty:
        db.attrs["spots"] = spots
        db.attrs["area"] = area
        db.attrs["gap"] = gap
        db.attrs["seed"] = seed
        db.close()

        return fname, n_samples

    rng = np.random.default_rng(seed)
    tile_size = np.asarray(area) / np.asarray(spots)

    filled_mat = rng.integers(2, size=(n_samples, spots[0], spots[1], spots[2]))
    mag_angle_mat = rng.random(size=(n_samples, spots[0] * spots[1] * spots[2]))

    for idx in tqdm(range(n_intv)):
        emp_x = spots[0] // 2 - 1
        emp_y = emp_x + 1
        s_x = emp_x * tile_size[0]
        s_y = emp_y * tile_size[1]

        filled_pos = [
            [i, j, k]
            for i in range(spots[0])
            for j in range(spots[1])
            for k in range(spots[2])
            if filled_mat[intv[0] + idx][i][j][k] == 1
            and (
                i < emp_x
                or i > emp_y
                or j < emp_x
                or j > emp_y
                or k < (spots[2] // 2 - 1)
                or k > spots[2] // 2
            )
        ]

        tiles = Tiles(
            n=len(filled_pos),
            size=tile_size,
            tile_type=2,
            M_rem=1.2 / (4 * np.pi * 1e-7),
            color=[1, 0, 0],
            mag_angle=[
                [np.pi / 2, 2 * np.pi * mag_angle_mat[idx + intv[0], i]]
                for i in range(len(filled_pos))
            ],
        )

        for i, pos in enumerate(filled_pos):
            pos_np = np.asarray(pos)
            if np.greater_equal(pos_np, spots).any():
                err_msg = f"Desired position {pos_np} is not in the grid!"
                raise ValueError(err_msg)
            tiles.offset = (np.around((pos_np + 0.5) * tile_size, decimals=9), i)

        x_eval = np.linspace(s_x + gap, s_y + tile_size[0] - gap, res[0])
        y_eval = np.linspace(s_x + gap, s_y + tile_size[1] - gap, res[1])

        if len(res) == 2:
            xv, yv = np.meshgrid(x_eval, y_eval)
            zv = np.zeros(res[0] * res[1]) + area[2] / 2

        elif len(res) == 3:
            # Pixel length in z-direction equal to x-direction
            s_z = (s_y - s_x) / res[0]
            z_eval = np.linspace(-s_z, s_z, res[2]) + area[2] / 2
            xv, yv, zv = np.meshgrid(x_eval, y_eval, z_eval)

        else:
            err_msg = "Only 2-D and 3-D magnetic field can be generated!"
            raise ValueError(err_msg)

        pts_eval = np.hstack([xv.reshape(-1, 1), yv.reshape(-1, 1), zv.reshape(-1, 1)])
        devnull = open("/dev/null", "w")  # noqa: PTH123, SIM115
        oldstdout_fno = os.dup(sys.stdout.fileno())
        os.dup2(devnull.fileno(), 1)
        _, h_out = run_simulation(tiles, pts_eval)
        os.dup2(oldstdout_fno, 1)

        # Tensor image with shape CxHxWxD [T]
        field = h_out.reshape((*res, 3))
        field = (
            field.transpose((3, 0, 1, 2))
            if len(res) == 3
            else field.transpose((2, 0, 1))
        )

        # Transpose the field to have 'ij' indexing
        if z_comp:
            field_ij = np.stack([field[0].T, field[1].T, field[2].T], axis=0)
        else:
            field_ij = np.stack([field[0].T, field[1].T], axis=0)
        db["field"][idx] = field_ij * 4 * np.pi * 1e-7

    db.close()


def create_db_mp(  # noqa: PLR0912
    data: str,
    datapath: Path | None = None,
    n_workers: int | None = None,
    **kwargs,
) -> None:
    if datapath is None:
        datapath = Path(__file__).parent.absolute() / ".." / ".." / ".." / "data"
    if not datapath.exists():
        datapath.mkdir(parents=True)
    kwargs["datapath"] = datapath

    if data == "magfield":
        target = db_magfield
    elif data == "magfield_symm":
        target = db_magfield_symm
    else:
        raise NotImplementedError

    db_name, n_tasks = target(**kwargs, empty=True)

    if n_workers is None:
        n_workers = cpu_count()
    intv = n_tasks // n_workers
    if n_tasks % n_workers > 0:
        intv += 1

    l_p = []
    for i in range(n_workers):
        end_intv = min((i + 1) * intv, n_tasks)
        kwargs["intv"] = [i * intv, end_intv]
        p = Process(target=target, kwargs=kwargs)
        p.daemon = True
        p.start()
        l_p.append(p)
        if end_intv == n_tasks:
            break

    try:
        for p in l_p:
            p.join()

    except KeyboardInterrupt:
        for p in l_p:
            p.terminate()

        path = datapath.glob("**/*")
        fnames = [
            x.name
            for x in path
            if x.is_file()
            and x.name[: len(db_name)] == db_name
            and x.name[:-3] != db_name
        ]

        for name in fnames:
            Path(datapath, name).unlink()

        Path(datapath, f"{db_name}.h5").unlink()
        sys.exit(130)

    path = datapath.glob("**/*")
    fnames = [
        x.name
        for x in path
        if x.is_file() and x.name[: len(db_name)] == db_name and x.name[:-3] != db_name
    ]

    with h5py.File(f"{datapath}/{db_name}.h5", mode="a") as db_t:
        for name in fnames:
            print(Path(datapath, name))
            with h5py.File(Path(datapath, name), mode="r") as db_s:
                intv = db_s.attrs["intv"]
                for key in db_s:
                    db_t[key][intv[0] : intv[1]] = db_s[key]
            Path(datapath, name).unlink()

    print("Database created with the following name:", db_name)
