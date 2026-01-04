<< 아키텍처 다이어그램 >> 넣기

### [테라폼 설치](https://developer.hashicorp.com/terraform/install) ###
mac 의 경우 아래의 명령어로 설치할 수 있다. 
```
brew tap hashicorp/tap
brew install hashicorp/tap/terraform
```

### VPC 생성 ###
테라폼으로 VPC 및 접속용 vs-code EC2 인스턴스를 생성한다.   
```
git pull https://github.com/gnosia93/training-on-eks.git
cd training-on-eks/tf
terraform init
terraform apply -auto-approve
```

* 생성된 인프라 확인
```
terraform output
```

### VPC 삭제 ###
생성된 VPC 와 관련 리소스를 모두 삭제한다.
```
terraform destroy --auto-approve
```
