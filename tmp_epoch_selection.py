from utilities.utils import epoch_selection

log_csv_path = "/NAS2020/Workspaces/DRLGroup/bohuang/bocode/safe-decision-diffuser/logs/tcdbc_dsrl/OfflineBallRun-v0/transformer-gw_1.2-cdp_0.2-CDFNormalizer-normret_True/300/2024-3-10-1/progress.csv"
save_epoch = 10
best_epoch = epoch_selection(log_csv_path, save_epoch)
print(best_epoch)

