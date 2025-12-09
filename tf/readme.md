<< 아키텍처 다이어그램 >> 넣기

위와 같은 아키텍처를 생성하기 아래 테라폼 명령어를 실행한다. 
```
cd training-on-eks/tf
terraform init

terraform apply -auto-approve
```


```
# Install VS Code Server (Code Server) - 설치하면 자동으로 systemctl 에 등록된다.
echo "install code-server ..."
curl -fsSL https://code-server.dev/install.sh | sh
sudo systemctl enable --now code-server@ec2-user

CONFIG_FILE="/home/ec2-user/.config/code-server/config.yaml"
sed -i 's/^\s*bind-addr:\s*127.0.0.1:8080/bind-addr: 0.0.0.0:8080/g' "$CONFIG_FILE"
sed -i 's/^\s*auth:\s*password/auth: none/g' "$CONFIG_FILE"

sudo systemctl restart code-server@ec2-user  
```
