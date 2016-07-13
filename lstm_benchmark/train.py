#Train
batch_size = 128
epochs = 1
history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    nb_epoch=epochs,
                    validation_data=(X_test, y_test),
                    callbacks=[checkpoint])