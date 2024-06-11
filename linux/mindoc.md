



```shell
/Library//LaunchDaemons/mindocd.plist

# 启动
sudo launchctl load -w /Library//LaunchDaemons/mindocd.plist

# 停止
launchctl unload -w /Library//LaunchDaemons/mindocd.plist

# 查看
sudo launchctl list | grep mindoc
```

