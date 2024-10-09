// src/pages/Livestream.js
import React from 'react';
import { Box, Image, Text } from '@chakra-ui/react';

const Viewer = () => {
  const streamUrl = `${import.meta.env.VITE_BACKEND_URL}/stream`;
  console.log(streamUrl);

  return (
    <Box
      display="flex"
      flexDirection="column"
      alignItems="center"
      justifyContent="center"
      minHeight="100vh"
      bg="gray.800"
      p={4}
    >
      <Text fontSize="2xl" color="white" mb={4}>
        Live Stream
      </Text>
      <Image
        src={streamUrl}
        alt="Live Stream"
        maxWidth="100%"
        borderRadius="md"
        boxShadow="lg"
      />
    </Box>
  );
};

export default Viewer;
