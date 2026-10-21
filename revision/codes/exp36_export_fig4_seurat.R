# Export the cells of the Figure 4 Seurat objects (notebook 13: X_root_integrated.rds for the
# scRNA-seq batch, P_root_integrated.rds for the GeneSys-generated batch) so that the Python
# port of the marker call can be validated on exactly the cells behind the published tables.
suppressMessages(library(SeuratObject)); suppressMessages(library(Matrix))
D <- "/resnick/groups/NolanLab/cwhsu/CW_data/genesys"
O <- "/resnick/groups/NolanLab/cwhsu/GeneSys_Revision/experiments/results/exp36_fig4_objects"
dir.create(O, showWarnings = FALSE, recursive = TRUE)
for (tag in c("P", "X")) {
  seu <- readRDS(file.path(D, paste0(tag, "_root_integrated.rds")))
  m <- tryCatch(GetAssayData(seu, slot = "data"), error = function(e) GetAssayData(seu, layer = "data"))
  cat(tag, dim(m), class(m), "\n")
  con <- file(file.path(O, paste0(tag, "_data_float32_genes_by_cells.bin")), "wb")
  for (a in seq(1, ncol(m), by = 1000)) {
    b <- min(a + 999, ncol(m))
    writeBin(as.numeric(as.matrix(m[, a:b])), con, size = 4)
  }
  close(con)
  write.csv(data.frame(cell = colnames(m), celltype_timebin = seu$celltype_timebin),
            file.path(O, paste0(tag, "_cells.csv")), row.names = FALSE)
  writeLines(rownames(m), file.path(O, paste0(tag, "_genes.txt")))
}
cat("[Exp 36] Done!\n")
