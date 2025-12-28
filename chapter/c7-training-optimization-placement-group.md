EKS 에서 Placement Group 및 Capacity Block 설정은 EC2 론치 템플릿을 사용한다.   

## 배치 그룹(Placement Group) ##
네트워크 지연 시간을 줄이려면, EKS 노드 그룹 생성 시 Placement Group을 적용해야 한다. 이는 EKS 노드가 생성될 때 도일 랙 또는 통신 홉을 최소화 하는 인접 랙 또는 근거리 랙에 서버를 배치하는 기능이다.  
```
aws ec2 create-placement-group \
    --group-name "deepspeed-placement-group" \
    --strategy cluster
```

### EC2 론치 템플릿 ###
```
cat <<'EOF' > launch-template-data.json
{
    "InstanceType": "p4d.24xlarge",
    "Placement": {
        "GroupName": "deepspeed-placement-group"
    },
//    "CapacityReservationSpecification": {
//        "CapacityReservationTarget": {
//            "CapacityReservationId": "cr-xxxxxxxxxxxxxxxxx"
//        }
//    },
    "BlockDeviceMappings": [
        {
            "DeviceName": "/dev/xvda",
            "Ebs": {
                "VolumeSize": 500,
                "VolumeType": "gp3"
            }
        }
    ],
    "TagSpecifications": [
        {
            "ResourceType": "instance",
            "Tags": [{ "Key": "Name", "Value": "eks-deepspeed-node" }]
        }
    ]
}
EOF

aws ec2 create-launch-template \
    --launch-template-name "deepspeed-launch-template" \
    --launch-template-data file://launch-template-data.json
```

### 노드 롤 ###
```
cat <<EOF > node-role-trust-relationship.json 
{
    "Version":"2012-10-17",		 	 	 
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "sts:AssumeRole"
            ],
            "Principal": {
                "Service": [
                    "ec2.amazonaws.com"
                ]
            }
        }
    ]
}
EOF

aws iam create-role --role-name trainig-on-eks-AmazonEKSNodeRole \
  --assume-role-policy-document file://"node-role-trust-relationship.json"

aws iam attach-role-policy --policy-arn arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy \
  --role-name trainig-on-eks-AmazonEKSNodeRole
aws iam attach-role-policy --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPullOnly \
  --role-name trainig-on-eks-AmazonEKSNodeRole
aws iam attach-role-policy --policy-arn arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy \
  --role-name trainig-on-eks-AmazonEKSNodeRole
aws iam attach-role-policy --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore \
  --role-name trainig-on-eks-AmazonEKSNodeRole
```

### EKS 노드그룹 ###
```
aws eks create-nodegroup \
    --cluster-name training-on-eks \
    --nodegroup-name "ng-deepspeed" \
    --launch-template name="deepspeed-launch-template",version=1 \
    --scaling-config minSize=2,maxSize=2,desiredSize=2 \
    --subnets "subnet-0e7be3e3155f668ed" "subnet-00b7e6cc786475a22" \
    --node-role "arn:aws:iam::499514681453:role/trainig-on-eks-AmazonEKSNodeRole"
```

### 참고 - 카펜터 Capacity Block ###
카펜터는 ODCR 과 Capacity Block 설정을 지원하지만 (아래 메뉴얼 참고), Placement Group 은 명시적으로 지원하지 않는다. 
* https://karpenter.sh/docs/tasks/odcrs/
* https://karpenter.sh/docs/concepts/nodeclasses/
* https://karpenter.sh/docs/concepts/nodepools/
  

## 레퍼런스 ##

* https://docs.aws.amazon.com/eks/latest/userguide/create-node-role.html
* https://docs.aws.amazon.com/cli/latest/reference/eks/create-nodegroup.html
* https://github.com/eksctl-io/eksctl/tree/main
  
