#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(stringr)
  library(jsonlite)
  library(scales)
})

`%||%` <- function(a, b) if (!is.null(a)) a else b

theme_clean <- function() {
  theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", size = 16, color = "#111827"),
      plot.subtitle = element_text(size = 11, color = "#4b5563"),
      axis.title = element_text(size = 11, color = "#1f2937"),
      axis.text = element_text(size = 10, color = "#111827"),
      panel.grid.minor = element_blank(),
      legend.position = "top",
      legend.title = element_text(size = 10),
      legend.text = element_text(size = 10)
    )
}

parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  cfg <- list(
    eval_dir = "outputs/full_task_eval_v1",
    blend_dir = "outputs/kcat_blend",
    ligase_train_dir = "outputs/ligase_multitask_v3_1",
    out_dir = "reports/ggplot2"
  )
  if (length(args) == 0) return(cfg)

  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    val <- if (i < length(args)) args[[i + 1]] else NA_character_
    if (key == "--eval-dir") cfg$eval_dir <- val
    if (key == "--blend-dir") cfg$blend_dir <- val
    if (key == "--ligase-train-dir") cfg$ligase_train_dir <- val
    if (key == "--out-dir") cfg$out_dir <- val
    i <- i + 2
  }
  cfg
}

safe_num <- function(x) {
  if (is.null(x)) return(NA_real_)
  y <- suppressWarnings(as.numeric(x))
  if (length(y) == 0) return(NA_real_)
  y[[1]]
}

extract_classification_metrics <- function(metrics_path, task_name) {
  if (!file.exists(metrics_path)) return(NULL)
  j <- fromJSON(metrics_path, simplifyVector = TRUE)

  tibble(
    task = task_name,
    accuracy = safe_num(j$accuracy %||% j$acc),
    precision = safe_num(j$precision %||% j$precision_macro),
    recall = safe_num(j$recall %||% j$recall_macro),
    f1 = safe_num(j$f1 %||% j$f1_macro %||% j$weighted_f1),
    roc_auc = safe_num(j$roc_auc %||% j$auc)
  )
}

plot_classification_summary <- function(eval_dir, out_dir) {
  metrics_csv <- file.path(eval_dir, "classification_metrics.csv")
  if (file.exists(metrics_csv)) {
    d0 <- suppressMessages(read_csv(metrics_csv, show_col_types = FALSE))
    if (nrow(d0) > 0) {
      d0 <- d0 %>%
        transmute(
          task = case_when(
            task == "ligase_identification" ~ "Ligase",
            task == "cofactor_atp_vs_nad" ~ "ATP/NAD",
            task == "solubility" ~ "Solubility",
            TRUE ~ task
          ),
          accuracy = suppressWarnings(as.numeric(accuracy)),
          f1 = suppressWarnings(as.numeric(f1)),
          roc_auc = suppressWarnings(as.numeric(roc_auc))
        )

      long_df <- d0 %>%
        pivot_longer(cols = c(accuracy, f1, roc_auc), names_to = "metric", values_to = "value") %>%
        filter(!is.na(value))
      if (nrow(long_df) == 0) return(FALSE)

      long_df$metric <- factor(long_df$metric, levels = c("accuracy", "f1", "roc_auc"))

      p <- ggplot(long_df, aes(x = metric, y = value, fill = task)) +
        geom_col(position = position_dodge(width = 0.75), width = 0.68, color = "white", linewidth = 0.2) +
        geom_text(
          aes(label = sprintf("%.3f", value)),
          position = position_dodge(width = 0.75),
          vjust = -0.35,
          size = 3.1,
          family = "sans"
        ) +
        scale_y_continuous(labels = label_number(accuracy = 0.01), limits = c(0, 1.05)) +
        scale_fill_manual(values = c("Ligase" = "#1d4ed8", "ATP/NAD" = "#0f766e", "Solubility" = "#b45309")) +
        labs(
          title = "Classification Metrics Overview",
          subtitle = "Higher is better (0-1)",
          x = NULL,
          y = "Score",
          fill = "Task"
        )
      p <- p + theme_clean()

      ggsave(file.path(out_dir, "classification_metrics_overview.png"), p, width = 10, height = 5.8, dpi = 320)
      return(TRUE)
    }
  }

  task_map <- c(
    ligase_identification = "Ligase",
    cofactor_atp_vs_nad = "ATP/NAD",
    solubility = "Solubility"
  )

  rows <- lapply(names(task_map), function(k) {
    p <- file.path(eval_dir, k, "metrics.json")
    extract_classification_metrics(p, task_map[[k]])
  })
  df <- bind_rows(rows)
  if (nrow(df) == 0) return(FALSE)

  long_df <- df %>%
    pivot_longer(cols = c(accuracy, precision, recall, f1, roc_auc), names_to = "metric", values_to = "value") %>%
    filter(!is.na(value))
  if (nrow(long_df) == 0) return(FALSE)

  long_df$metric <- factor(long_df$metric, levels = c("accuracy", "precision", "recall", "f1", "roc_auc"))

  p <- ggplot(long_df, aes(x = metric, y = value, fill = task)) +
    geom_col(position = position_dodge(width = 0.75), width = 0.68, color = "white", linewidth = 0.2) +
    geom_text(
      aes(label = sprintf("%.3f", value)),
      position = position_dodge(width = 0.75),
      vjust = -0.35,
      size = 3.1,
      family = "sans"
    ) +
    scale_y_continuous(labels = label_number(accuracy = 0.01), limits = c(0, 1.05)) +
    scale_fill_manual(values = c("Ligase" = "#1d4ed8", "ATP/NAD" = "#0f766e", "Solubility" = "#b45309")) +
    labs(
      title = "Classification Metrics Overview",
      subtitle = "Higher is better (0-1)",
      x = NULL,
      y = "Score",
      fill = "Task"
    )
  p <- p + theme_clean()

  ggsave(file.path(out_dir, "classification_metrics_overview.png"), p, width = 10, height = 5.8, dpi = 320)
  TRUE
}

plot_train_history <- function(train_dir, out_dir) {
  hist_path <- file.path(train_dir, "train_history.csv")
  if (!file.exists(hist_path)) return(FALSE)

  df <- suppressMessages(read_csv(hist_path, show_col_types = FALSE))
  if (!("epoch" %in% colnames(df))) return(FALSE)

  metric_candidates <- c("val_ec_acc", "val_ec_f1", "val_sub_f1", "val_metal_f1", "val_score")
  metric_cols <- metric_candidates[metric_candidates %in% colnames(df)]

  p1 <- ggplot(df, aes(x = epoch, y = train_loss)) +
    geom_line(color = "#dc2626", linewidth = 1) +
    geom_point(color = "#dc2626", size = 1.8) +
    labs(title = "Ligase Multi-task Training Loss", x = "Epoch", y = "Train Loss")
  p1 <- p1 + theme_clean()

  ggsave(file.path(out_dir, "ligase_train_loss_curve.png"), p1, width = 8.5, height = 4.8, dpi = 320)

  if (length(metric_cols) > 0) {
    d2 <- df %>%
      select(epoch, all_of(metric_cols)) %>%
      pivot_longer(cols = all_of(metric_cols), names_to = "metric", values_to = "value")

    p2 <- ggplot(d2, aes(x = epoch, y = value, color = metric)) +
      geom_line(linewidth = 1) +
      geom_point(size = 1.5) +
      scale_color_manual(values = c("#2563eb", "#0f766e", "#9333ea", "#b45309", "#e11d48")) +
      labs(title = "Validation Metrics by Epoch", x = "Epoch", y = "Score", color = "Metric")
    p2 <- p2 + theme_clean()

    ggsave(file.path(out_dir, "ligase_val_metrics_curve.png"), p2, width = 9.6, height = 5.4, dpi = 320)
  }

  TRUE
}

plot_kcat_blend_scatter <- function(blend_dir, out_dir) {
  oof_path <- file.path(blend_dir, "blend", "oof_predictions.csv")
  if (!file.exists(oof_path)) return(FALSE)

  df <- suppressMessages(read_csv(oof_path, show_col_types = FALSE))
  y_col <- if ("y_true" %in% colnames(df)) "y_true" else if ("y" %in% colnames(df)) "y" else NA_character_
  p_col <- if ("y_pred_blend" %in% colnames(df)) "y_pred_blend" else if ("y_pred" %in% colnames(df)) "y_pred" else NA_character_
  if (is.na(y_col) || is.na(p_col)) return(FALSE)

  x <- df[[y_col]]
  y <- df[[p_col]]
  rr <- suppressWarnings(cor(x, y, use = "complete.obs"))

  lim_low <- min(c(x, y), na.rm = TRUE)
  lim_high <- max(c(x, y), na.rm = TRUE)

  p <- ggplot(df, aes_string(x = y_col, y = p_col)) +
    geom_point(color = "#2563eb", alpha = 0.65, size = 2) +
    geom_abline(slope = 1, intercept = 0, color = "#64748b", linetype = "dashed", linewidth = 0.8) +
    geom_smooth(method = "lm", se = TRUE, color = "#dc2626", linewidth = 1) +
    coord_equal(xlim = c(lim_low, lim_high), ylim = c(lim_low, lim_high)) +
    annotate("label", x = lim_low + 0.12 * (lim_high - lim_low), y = lim_high - 0.06 * (lim_high - lim_low),
             label = sprintf("Pearson R = %.3f", rr), fill = "white", color = "#0f172a") +
    labs(
      title = "kcat Blend OOF Scatter (log_kcat)",
      subtitle = "Blue: OOF predictions, Red: linear fit, Dashed: y=x",
      x = "True log_kcat",
      y = "Predicted log_kcat"
    )
  p <- p + theme_clean()

  ggsave(file.path(out_dir, "kcat_blend_oof_scatter.png"), p, width = 7.2, height = 6.6, dpi = 320)
  TRUE
}

plot_confusion_matrices <- function(eval_dir, out_dir) {
  pred_files <- list.files(eval_dir, pattern = "val_predictions\\.csv$", recursive = TRUE, full.names = TRUE)
  made_any <- FALSE
  for (f in pred_files) {
    d <- suppressMessages(read_csv(f, show_col_types = FALSE))
    y_true_col <- if ("y_true" %in% colnames(d)) "y_true" else if ("label" %in% colnames(d)) "label" else NA_character_
    y_pred_col <- if ("y_pred" %in% colnames(d)) "y_pred" else if ("pred" %in% colnames(d)) "pred" else NA_character_
    if (is.na(y_true_col) || is.na(y_pred_col)) next

    cm <- d %>%
      transmute(True = as.factor(.data[[y_true_col]]), Predicted = as.factor(.data[[y_pred_col]])) %>%
      count(True, Predicted, name = "Count") %>%
      complete(True, Predicted, fill = list(Count = 0))

    p <- ggplot(cm, aes(x = Predicted, y = True, fill = Count)) +
      geom_tile(color = "white", linewidth = 0.25) +
      geom_text(aes(label = Count), size = 3) +
      scale_fill_gradient(low = "#eff6ff", high = "#1d4ed8") +
      labs(
        title = paste0("Confusion Matrix: ", basename(dirname(f))),
        x = "Predicted label",
        y = "True label"
      )
    p <- p + theme_clean() + theme(panel.grid = element_blank())

    out_name <- paste0(basename(dirname(f)), "_confusion_ggplot.png")
    ggsave(file.path(out_dir, out_name), p, width = 6.6, height = 5.4, dpi = 320)
    made_any <- TRUE
  }
  if (made_any) return(TRUE)

  files <- list.files(eval_dir, pattern = "confusion.*\\.csv$", recursive = TRUE, full.names = TRUE)
  if (length(files) == 0) return(FALSE)

  made_any <- FALSE
  for (f in files) {
    d <- suppressMessages(read_csv(f, show_col_types = FALSE))
    if (ncol(d) < 2) next

    first_col <- colnames(d)[1]
    if (!str_detect(tolower(first_col), "true|label|class|actual|^x$")) {
      next
    }

    mat <- d %>%
      pivot_longer(cols = -all_of(first_col), names_to = "Predicted", values_to = "Count") %>%
      rename(True = all_of(first_col))

    p <- ggplot(mat, aes(x = Predicted, y = True, fill = Count)) +
      geom_tile(color = "white", linewidth = 0.25) +
      geom_text(aes(label = Count), size = 3) +
      scale_fill_gradient(low = "#eff6ff", high = "#1d4ed8") +
      labs(
        title = paste0("Confusion Matrix: ", basename(dirname(f))),
        x = "Predicted label",
        y = "True label"
      )
    p <- p + theme_clean() + theme(axis.text.x = element_text(angle = 30, hjust = 1), panel.grid = element_blank())

    out_name <- paste0(tools::file_path_sans_ext(basename(f)), "_ggplot.png")
    ggsave(file.path(out_dir, out_name), p, width = 7.8, height = 6.0, dpi = 320)
    made_any <- TRUE
  }

  made_any
}

main <- function() {
  cfg <- parse_args()
  dir.create(cfg$out_dir, recursive = TRUE, showWarnings = FALSE)

  done <- c(
    classification = plot_classification_summary(cfg$eval_dir, cfg$out_dir),
    train_history = plot_train_history(cfg$ligase_train_dir, cfg$out_dir),
    kcat_scatter = plot_kcat_blend_scatter(cfg$blend_dir, cfg$out_dir),
    confusion = plot_confusion_matrices(cfg$eval_dir, cfg$out_dir)
  )

  summary_path <- file.path(cfg$out_dir, "plot_generation_summary.json")
  write_json(
    list(
      eval_dir = cfg$eval_dir,
      blend_dir = cfg$blend_dir,
      ligase_train_dir = cfg$ligase_train_dir,
      out_dir = cfg$out_dir,
      generated = done,
      generated_at = as.character(Sys.time())
    ),
    summary_path,
    pretty = TRUE,
    auto_unbox = TRUE
  )

  message("[Done] Output dir: ", cfg$out_dir)
  message("[Done] Summary: ", summary_path)
}

main()
