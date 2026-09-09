rule make_cell_images_bbox:
    """Patched copy of starcall-workflow's `rule make_cell_images`
    (workflow/rules/phenotyping.smk): crops every segmented cell out of one
    tile's whole stitched phenotype image + mask, using each cell's
    bbox-midpoint as its crop center (bbox_x1/y1/x2/y2 -- the real schema
    has no xpos/ypos) instead of the upstream rule's own centroid columns.
    Output shape/semantics otherwise identical: (num_cells, num_channels,
    window, window) crop stack + matching (num_cells, window, window)
    uint8 mask-crop stack, one file pair per tile, mask label i+1 == cell
    table's i-th row (0-based) -- unchanged from the real rule.
    See docs/architecture.md decision 17.
    """
    input:
        image = get_phenotyping_pt,
        cells = phenotyping_dir + '{path}/{segmentation_type}_mask.tif',
        cell_table = phenotyping_dir + '{path}/{segmentation_type}.csv',
    output:
        cell_images = phenotyping_dir + '{path}/{segmentation_type}_crops_{window,\d+}.tif',
        mask_images = phenotyping_dir + '{path}/{segmentation_type}_mask_crops_{window,\d+}.tif',
    resources:
        mem_mb = lambda wildcards, input: input.size_mb * 2.5 + 5000
    run:
        import numpy as np
        import tifffile
        import pandas

        cell_table = pandas.read_csv(input.cell_table, index_col=0)

        if len(cell_table.index) == 0:
            os.system('touch {}'.format(output.cell_images))
            os.system('touch {}'.format(output.mask_images))
        else:
            cells = tifffile.imread(input.cells)
            image = tifffile.imread(input.image)
            image = image.reshape(-1, *image.shape[2:])
            debug(image.shape)

            window = int(wildcards.window)
            window_low = window // 2
            window_high = window - window_low

            cell_images = np.zeros((len(cell_table), image.shape[0], window, window), image.dtype)
            mask_images = np.zeros((len(cell_table), window, window), dtype=np.uint8)

            for i, cell_index in enumerate(cell_table.index):
                debug(cell_index)
                x1_bbox = int(cell_table['bbox_x1'][cell_index])
                y1_bbox = int(cell_table['bbox_y1'][cell_index])
                x2_bbox = int(cell_table['bbox_x2'][cell_index])
                y2_bbox = int(cell_table['bbox_y2'][cell_index])
                centroid = (x1_bbox + x2_bbox) // 2, (y1_bbox + y2_bbox) // 2

                x1, x2, y1, y2 = centroid[0] - window_low, centroid[0] + window_high, centroid[1] - window_low, centroid[1] + window_high
                x1, x2, y1, y2 = max(0, x1), min(cells.shape[0], x2), max(0, y1), min(cells.shape[1], y2)
                subset = image[:, x1:x2, y1:y2]
                mask = cells[x1:x2, y1:y2] == i + 1
                x1, x2 = window_low - (centroid[0] - x1), window_low + (x2 - centroid[0])
                y1, y2 = window_low - (centroid[1] - y1), window_low + (y2 - centroid[1])
                cell_images[i, :, x1:x2, y1:y2] = subset
                mask_images[i, x1:x2, y1:y2] = mask

            tifffile.imwrite(output.cell_images, cell_images)
            tifffile.imwrite(output.mask_images, mask_images)

# Both rules declare the exact same output path -- disambiguate in our
# favor. Standard Snakemake mechanism for "prefer my rule over an existing
# one with the same output"; more robust than hoping a same-named
# redeclaration silently wins (a true duplicate rule name raises
# WorkflowError/AmbiguousRuleException) or relying on `module`/`use rule
# ... with:` (version-sensitive, not confirmed to support full run:-block
# replacement).
ruleorder: make_cell_images_bbox > make_cell_images
