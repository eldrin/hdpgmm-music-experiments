from typing import Any, Callable
import argparse
from pathlib import Path
from itertools import groupby
from dataclasses import dataclass


from tqdm import tqdm


@dataclass
class Interaction:
    """
    it contains each user's interaction record
    with a certain item. (i.e., (userid, itemid, intensity))

    Attributes:
        user (Any): user id
        item (Any): item id
        intensity (int): measure indicating how many or much this user
                         interacted with this item
    """
    user: Any
    item: Any
    intensity: int


@dataclass
class InteractionDataset:
    """
    it contains interactions and unique user and item list.

    Attributs:
        interactions (list[Interaction]): list of interactions
        users (list[Any]): unique user list
        items (list[Any]): unique item list
    """
    interactions: list[Interaction]
    users: list[Any]
    items: list[Any]

    def __len__(self):
        return len(self.interactions)


def _filter_triplets(
    triplets: list[Interaction],
    threshold: int,
    key_func: Callable = lambda x: x.user
) -> list[Interaction]:
    """
    filters triplets on the axis given by `key_func` and `threshold`.

    Args:
        triplets: list of interactions to be filtered.
        threshold: entity specified by `key_func` having frequency smaller
                   than this value will be filtered.
        key_func: specifies the entity to be filtered.

    Returns:
        filtered list of interactions
    """
    triplets_filtered = []
    for _, group in groupby(sorted(triplets, key=key_func), key=key_func):
        g = list(group)
        if len(g) < threshold:
            continue
        triplets_filtered.extend(g)
    return triplets_filtered


def freq_based_filter(
    triplets: list[Interaction],
    user_thresh: int = 5,
    item_thresh: int = 0,
    num_iters: int = 1,
    verbose: bool = False
) -> InteractionDataset:
    """
    it filters out users and items based on their frequency.

    Args:
        triplets: the list of raw interaction
        user_thresh: threshold for user frequency. If the number of
                     unique items an user interacted is smaller than
                     it, those are not included (filtered).
        item_thresh: threshold for items. works same as above.
        num_iter: the number of filtering routine. As the change in
                  one filtering step (i.e., user filtering or item filtering)
                  changes the dataset, so we iterate the process.

    Returns:
        the processed dataset contains triplets and user/item list
    """
    # filter interactions
    tmp = triplets
    with tqdm(total=num_iters, ncols=80, disable=not verbose) as prog:
        for _ in range(num_iters):
            tmp = _filter_triplets(tmp, item_thresh, lambda x: x.item)
            tmp = _filter_triplets(tmp, user_thresh, lambda x: x.user)
            prog.update()

    # build dataset
    triplets = []
    users, items = {}, {}
    users_list, items_list = [], []
    with tqdm(total=len(tmp), ncols=80, disable=not verbose) as prog:
        for row in tmp:
            if row.user not in users:
                users[row.user] = len(users)
                users_list.append(row.user)
            if row.item not in items:
                items[row.item] = len(items)
                items_list.append(row.item)
            triplets.append(
                Interaction(users[row.user], items[row.item], row.intensity)
            )
            prog.update()
    return InteractionDataset(triplets, users_list, items_list)


def load_raw_triplets(
    path: str,
    delimiter: str = '\t'
) -> list[Interaction]:
    """
    read triplets from disk and format it to the list of interactions

    Args:
        path: string contains path of the raw dataset. it assumes text file
              containing triplets per each line
        delimiter: string that separate the `columns` of triplet

    Returns:
        list of interaction after loading
    """
    with Path(path).open('r') as fp:
        triplets = []
        for line in fp:
            u, i, c = line.replace('\n', '').split(delimiter)
            triplets.append(Interaction(u, i, int(c)))
    return triplets


def save(
    dataset: InteractionDataset,
    out_path: str,
    out_name: str,
    delimiter: str = '\t',
    verbose: bool = False
) -> None:
    """
    saves the filtered triplets into the disk

    Args:
        dataset: processed interaction dataset
        out_path: directory where the output is stored
        out_name: filename (stem) that the processed dataset will use
    """
    with (Path(out_path) / f'{out_name}.txt').open('w') as fp:
        with tqdm(total=len(dataset), ncols=80, disable=not verbose) as prog:
            for row in dataset.interactions:
                user = dataset.users[row.user]
                item = dataset.items[row.item]
                line = delimiter.join([user, item, f'{row.intensity:d}'])
                fp.write(line + '\n')
                prog.update()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="filter_echonest",
        description=(
            "filter triplets based on user and item frequencies."
        )
    )

    parser.add_argument("triplet_path", type=str,
                        help="path where fitted feature learner model is located")
    parser.add_argument("out_path", type=str,
                        help="root directory where fitted model is stored")

    parser.add_argument("--out-name", type=str, default='filtered_triplets',
                        help="filename to be used for the processed data saved.")
    parser.add_argument("--user-thresh", type=int, default=5,
                        help="threshold of the user frequency to filter user inactive")
    parser.add_argument("--item-thresh", type=int, default=0,
                        help="threshold of the item frequency to filter item inactive")
    parser.add_argument("--num-iters", type=int, default=1,
                        help="the number of iteration for the filtering process")
    parser.add_argument("--delimiter", type=str, default="\t",
                        help="delimiter that separates 'columns' of the triplets")
    parser.add_argument('--verbose', default=True,
                        action=argparse.BooleanOptionalAction,
                        help="set verbosity")
    return parser.parse_args()


def main() -> None:
    """
    """
    # parse input arguments
    args = parse_arguments()

    # load the raw triplets from disk
    raw_triplets = load_raw_triplets(args.triplet_path)

    # process them
    result = freq_based_filter(raw_triplets,
                               args.user_thresh,
                               args.item_thresh,
                               args.num_iters,
                               verbose=args.verbose)

    # save to disk
    save(result,
         args.out_path,
         args.out_name,
         verbose = args.verbose)


if __name__ == "__main__":
    main()
