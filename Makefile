ub:
	docker compose up --build

u:
	docker compose up

d:
	docker compose down

gpu:
	docker compose -f compose.yaml -f compose.gpu.yaml up --build
