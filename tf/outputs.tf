output "instance_public_dns" {
  value       = aws_instance.graviton_box.public_dns
  description = "SSH 및 VS Code Server 접속을 위한 EC2 인스턴스의 공인 IP 주소"
}

output "vscode_url" {
    value = "http://${aws_instance.graviton_box.public_dns}:8080"
    description = "브라우저에서 VS Code 서버에 접속할 수 있는 URL (PW: 'password' by default)"
}

output "public_subnet" {
    description = "퍼블릭 서브넷 ID 목록"
    value       = aws_subnet.public[*].id
}

output "private_subnet" {
    description = "프라이빗 서브넷 ID 목록"
    value       = [for subnet in aws_subnet.private : subnet.id]
}

