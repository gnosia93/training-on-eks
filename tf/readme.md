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
```
[결과]
```
Initializing the backend...
Initializing provider plugins...
- Finding latest version of hashicorp/aws...
- Finding latest version of hashicorp/http...
- Installing hashicorp/aws v6.27.0...
- Installed hashicorp/aws v6.27.0 (signed by HashiCorp)
- Installing hashicorp/http v3.5.0...
- Installed hashicorp/http v3.5.0 (signed by HashiCorp)
Terraform has created a lock file .terraform.lock.hcl to record the provider
selections it made above. Include this file in your version control repository
so that Terraform can guarantee to make the same selections by default when
you run "terraform init" in the future.

Terraform has been successfully initialized!

You may now begin working with Terraform. Try running "terraform plan" to see
any changes that are required for your infrastructure. All Terraform commands
should now work.

If you ever set or change modules or backend configuration for Terraform,
rerun this command to reinitialize your working directory. If you forget, other
commands will detect it and remind you to do so if necessary.
```
VPC 를 생성한다. 
```
terraform apply -auto-approve
```

### VPC 확인 ###
```
terraform output
```

### VPC 삭제 ###
워크샵과 관련된 리소스를 모두 삭제한다.
```
terraform destroy --auto-approve
```

### 생성되는 리소스 ###

* VPC
* Subnets (Public / Private)
* Graviton EC2 for vs-code
* Security Groups
* Fsx for Lustre
* S3 bucket 
