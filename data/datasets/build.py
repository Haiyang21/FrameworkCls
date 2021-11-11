from data.datasets.expression import ExpressionDataset


META_DATASET = {
    'ExpressionDataset': ExpressionDataset,
}


def build_dataset(cfg, phase, transform=None):
    meta_dataset = META_DATASET[cfg.DATASET.DATASET_NAME]
    return meta_dataset(cfg, phase, transform)


