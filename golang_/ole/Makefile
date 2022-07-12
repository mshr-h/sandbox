SRC=main.go
GOOS=windows
GOARCH=amd64

all: $(SRC)
	GOOS=$(GOOS) GOARCH=$(GOARCH) go build $(SRC)

clean:
	rm main.exe