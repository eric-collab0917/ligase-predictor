#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
})

args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(flag, default = NULL) {
  idx <- match(flag, args)
  if (is.na(idx) || idx == length(args)) {
    return(default)
  }
  args[[idx + 1]]
}

slugify <- function(x) {
  y <- tolower(x)
  y <- gsub("[^a-z0-9]+", "_", y)
  y <- gsub("_+", "_", y)
  y <- gsub("^_|_$", "", y)
  y
}

pr_curve_df <- function(y_true, scores, label_name) {
  pos_total <- sum(y_true == 1)
  if (pos_total == 0) {
    return(data.frame())
  }
  ord <- order(scores, decreasing = TRUE)
  y <- y_true[ord]
  tp <- cumsum(y == 1)
  fp <- cumsum(y == 0)
  precision <- tp / pmax(tp + fp, 1)
  recall <- tp / pos_total
  data.frame(
    recall = c(0, recall),
    precision = c(1, precision),
    label = label_name
  )
}

mat_to_long <- function(mat, labels, panel_name) {
  d <- as.data.frame(as.table(mat), stringsAsFactors = FALSE)
  names(d) <- c("y", "x", "value")
  d$y <- factor(labels[d$y], levels = rev(labels))
  d$x <- factor(labels[d$x], levels = labels)
  d$panel <- panel_name
  d
}

theme_course <- function() {
  theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", colour = "#0f172a"),
      plot.subtitle = element_text(colour = "#475569"),
      axis.title = element_text(colour = "#0f172a"),
      axis.text = element_text(colour = "#0f172a"),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_line(colour = "#e2e8f0"),
      panel.grid.major.y = element_line(colour = "#e2e8f0"),
      strip.text = element_text(face = "bold", colour = "#0f172a"),
      legend.title = element_text(colour = "#0f172a"),
      legend.text = element_text(colour = "#0f172a")
    )
}

csv_path <- get_arg("--csv")
history_path <- get_arg("--train-history")
per_label_path <- get_arg("--per-label")
pred_path <- get_arg("--predictions")
sweep_path <- get_arg("--threshold-sweep")
outdir <- get_arg("--outdir")

if (is.null(csv_path) || is.null(history_path) || is.null(per_label_path) || is.null(pred_path) || is.null(sweep_path) || is.null(outdir)) {
  stop("Missing required arguments.")
}

dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

raw_df <- read.csv(csv_path, stringsAsFactors = FALSE)
hist_df <- read.csv(history_path, stringsAsFactors = FALSE)
per_label_df <- read.csv(per_label_path, stringsAsFactors = FALSE)
pred_df <- read.csv(pred_path, stringsAsFactors = FALSE)
sweep_df <- read.csv(sweep_path, stringsAsFactors = FALSE)

label_tokens <- unlist(strsplit(raw_df$subcellular_labels, ";", fixed = TRUE))
label_tokens <- trimws(label_tokens)
label_tokens <- label_tokens[!(tolower(label_tokens) %in% c("", "nan", "none", "null"))]
freq_df <- as.data.frame(table(label_tokens), stringsAsFactors = FALSE)
names(freq_df) <- c("label", "count")
freq_df <- freq_df[order(freq_df$count, decreasing = FALSE), ]
freq_df$label <- factor(freq_df$label, levels = freq_df$label)

p_freq <- ggplot(freq_df, aes(x = count, y = label)) +
  geom_col(fill = "#0f766e", width = 0.72) +
  geom_text(aes(label = count), hjust = -0.15, size = 3.6, colour = "#0f172a") +
  scale_x_continuous(expand = expansion(mult = c(0, 0.08))) +
  labs(
    title = "Subcellular Label Frequency",
    x = "Positive samples",
    y = NULL
  ) +
  theme_course()
ggsave(file.path(outdir, "fig_label_frequency.png"), p_freq, width = 9.5, height = 5.8, dpi = 240)

loss_df <- data.frame(
  epoch = c(hist_df$epoch, hist_df$epoch),
  value = c(hist_df$train_loss, hist_df$loss),
  series = c(rep("Train loss", nrow(hist_df)), rep("Val loss", nrow(hist_df))),
  panel = "Loss"
)
f1_df <- data.frame(
  epoch = c(hist_df$epoch, hist_df$epoch),
  value = c(hist_df$micro_f1, hist_df$macro_f1),
  series = c(rep("Micro F1", nrow(hist_df)), rep("Macro F1", nrow(hist_df))),
  panel = "Validation F1"
)
thr_df <- data.frame(
  epoch = hist_df$epoch,
  value = hist_df$threshold,
  series = "Threshold",
  panel = "Best threshold by epoch"
)
curve_df <- rbind(loss_df, f1_df, thr_df)

p_curves <- ggplot(curve_df, aes(x = epoch, y = value, colour = series)) +
  geom_line(linewidth = 1.1) +
  facet_wrap(~panel, scales = "free_y", nrow = 1) +
  scale_colour_manual(values = c("Train loss" = "#0f766e", "Val loss" = "#ea580c", "Micro F1" = "#2563eb", "Macro F1" = "#e11d48", "Threshold" = "#475569")) +
  labs(
    title = "Training Dynamics",
    x = "Epoch",
    y = NULL,
    colour = NULL
  ) +
  theme_course() +
  theme(legend.position = "bottom")
ggsave(file.path(outdir, "fig_training_curves.png"), p_curves, width = 15.5, height = 4.8, dpi = 240)

per_label_df <- per_label_df[order(per_label_df$f1, per_label_df$support), ]
per_label_df$label_display <- factor(
  paste0(per_label_df$label, " (n=", per_label_df$support, ")"),
  levels = paste0(per_label_df$label, " (n=", per_label_df$support, ")")
)

p_f1 <- ggplot(per_label_df, aes(x = f1, y = label_display, fill = support)) +
  geom_col(width = 0.72) +
  geom_text(aes(label = sprintf("%.2f", f1)), hjust = -0.15, size = 3.4, colour = "#0f172a") +
  scale_x_continuous(limits = c(0, 1.02), expand = expansion(mult = c(0, 0.02))) +
  scale_fill_gradient(low = "#99f6e4", high = "#0f766e") +
  labs(
    title = "Per-label F1 with Validation Support",
    x = "F1",
    y = NULL,
    fill = "Support"
  ) +
  theme_course()
ggsave(file.path(outdir, "fig_per_label_f1_support.png"), p_f1, width = 10.8, height = 6.2, dpi = 240)

ovr_rows <- do.call(rbind, lapply(seq_len(nrow(per_label_df)), function(i) {
  row <- per_label_df[i, ]
  data.frame(
    label = row$label,
    true_class = factor(c("True 0", "True 0", "True 1", "True 1"), levels = c("True 1", "True 0")),
    pred_class = factor(c("Pred 0", "Pred 1", "Pred 0", "Pred 1"), levels = c("Pred 0", "Pred 1")),
    value = c(row$tn, row$fp, row$fn, row$tp),
    f1 = row$f1
  )
}))

p_ovr <- ggplot(ovr_rows, aes(x = pred_class, y = true_class, fill = value)) +
  geom_tile(colour = "white", linewidth = 0.4) +
  geom_text(aes(label = value), size = 3.2, colour = "#0f172a", fontface = "bold") +
  scale_fill_gradient(low = "#e0f2fe", high = "#1d4ed8") +
  facet_wrap(~label, ncol = 4) +
  labs(
    title = "One-vs-rest Confusion Matrices",
    x = NULL,
    y = NULL,
    fill = "Count"
  ) +
  theme_course() +
  theme(panel.grid = element_blank())
ggsave(file.path(outdir, "fig_ovr_confusion_grid.png"), p_ovr, width = 15.5, height = 10.5, dpi = 240)

true_cols <- grep("^true__", names(pred_df), value = TRUE)
prob_cols <- grep("^prob__", names(pred_df), value = TRUE)
slug_map <- setNames(per_label_df$label, slugify(per_label_df$label))

pr_rows <- list()
for (true_col in true_cols) {
  slug <- sub("^true__", "", true_col)
  prob_col <- paste0("prob__", slug)
  if (!(prob_col %in% prob_cols)) {
    next
  }
  label_name <- slug_map[[slug]]
  if (is.null(label_name)) {
    label_name <- slug
  }
  pr_rows[[length(pr_rows) + 1]] <- pr_curve_df(pred_df[[true_col]], pred_df[[prob_col]], label_name)
}
pr_df <- do.call(rbind, pr_rows)

p_pr <- ggplot(pr_df, aes(x = recall, y = precision, colour = label)) +
  geom_line(linewidth = 1.0) +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1.02)) +
  labs(
    title = "Precision-Recall Curves by Label",
    x = "Recall",
    y = "Precision",
    colour = NULL
  ) +
  theme_course() +
  theme(legend.position = "bottom", legend.text = element_text(size = 8))
ggsave(file.path(outdir, "fig_precision_recall_curves.png"), p_pr, width = 9.6, height = 7.0, dpi = 240)

sweep_long <- rbind(
  data.frame(threshold = sweep_df$threshold, value = sweep_df$micro_f1, metric = "Micro F1", panel = "F1 vs threshold"),
  data.frame(threshold = sweep_df$threshold, value = sweep_df$macro_f1, metric = "Macro F1", panel = "F1 vs threshold"),
  data.frame(threshold = sweep_df$threshold, value = sweep_df$samples_f1, metric = "Samples F1", panel = "F1 vs threshold"),
  data.frame(threshold = sweep_df$threshold, value = sweep_df$mean_pred_labels, metric = "Predicted labels", panel = "Predicted labels per sample")
)
best_thr <- sweep_df$threshold[which.max(sweep_df$micro_f1)]

p_sweep <- ggplot(sweep_long, aes(x = threshold, y = value, colour = metric)) +
  geom_line(linewidth = 1.1) +
  geom_vline(xintercept = best_thr, linetype = "dashed", colour = "#475569") +
  facet_wrap(~panel, scales = "free_y", nrow = 1) +
  scale_colour_manual(values = c("Micro F1" = "#2563eb", "Macro F1" = "#e11d48", "Samples F1" = "#0f766e", "Predicted labels" = "#ea580c")) +
  labs(
    title = "Threshold Sweep",
    x = "Global threshold",
    y = NULL,
    colour = NULL
  ) +
  theme_course() +
  theme(legend.position = "bottom")
ggsave(file.path(outdir, "fig_threshold_sweep.png"), p_sweep, width = 13.6, height = 4.8, dpi = 240)

label_names <- per_label_df$label
true_mat <- as.matrix(pred_df[grep("^true__", names(pred_df), value = TRUE)])
pred_mat <- as.matrix(pred_df[grep("^pred__", names(pred_df), value = TRUE)])
true_co <- t(true_mat) %*% true_mat
pred_co <- t(pred_mat) %*% pred_mat
delta_co <- pred_co - true_co

co_df <- rbind(
  mat_to_long(true_co, label_names, "True co-occurrence"),
  mat_to_long(pred_co, label_names, "Predicted co-occurrence"),
  mat_to_long(delta_co, label_names, "Prediction minus truth")
)

p_co <- ggplot(co_df, aes(x = x, y = y, fill = value)) +
  geom_tile(colour = "white", linewidth = 0.2) +
  facet_wrap(~panel, nrow = 1) +
  scale_fill_gradient2(low = "#2563eb", mid = "#f8fafc", high = "#dc2626") +
  labs(
    title = "Label Co-occurrence Structure",
    x = NULL,
    y = NULL,
    fill = "Count"
  ) +
  theme_course() +
  theme(
    panel.grid = element_blank(),
    axis.text.x = element_text(angle = 50, hjust = 1, size = 8),
    axis.text.y = element_text(size = 8)
  )
ggsave(file.path(outdir, "fig_label_cooccurrence.png"), p_co, width = 18.5, height = 5.8, dpi = 240)

cat("[Done] Plots saved to:", outdir, "\n")
